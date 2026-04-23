"""
Flask веб-сервис для GuitarVision.
Транслирует видео с камеры и показывает анализ хвата.
"""

from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import threading
import time
import sys
import os
import json
from datetime import datetime

from grip_classifier import PickGripClassifier, GripType
from hand_tracking import HandTracker

app = Flask(__name__)

# Глобальные переменные для обмена данными между потоками
frame_lock = threading.Lock()
current_analysis = None
analysis_timestamp = 0
is_running = False
tolerance = 0.15  # По умолчанию 15% отклонения

# Инициализация камеры и классификатора
camera = None
tracker = None
classifier = None


def init_components(reference_image=None):
    """Инициализация камеры и классификатора."""
    global camera, tracker, classifier, tolerance
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Настраиваем камеру для минимальной задержки
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    tracker = HandTracker()
    
    if reference_image:
        classifier = PickGripClassifier(reference_image_path=reference_image, tolerance=tolerance)
    else:
        classifier = PickGripClassifier(tolerance=tolerance)
    
    return True


def generate_frames():
    """Генератор кадров для MJPEG стриминга с синхронным анализом."""
    global current_analysis, analysis_timestamp, is_running
    
    is_running = True
    
    while is_running:
        # Читаем кадр с камеры (CAP_PROP_BUFFERSIZE=1 уже обеспечивает свежий кадр)
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        # Зеркальное отражение (flip) для естественного отображения
        frame = cv2.flip(frame, 1)
        
        # Отрисовка landmarks и детекция (один вызов MediaPipe)
        frame = tracker.find_hands(frame, draw=True)
        
        # Получаем кэшированные landmarks (без повторной детекции)
        landmarks = tracker.get_cached_landmarks()
        
        if landmarks:
            analysis = classifier.analyze_grip(landmarks)
            
            # Обновляем глобальный анализ синхронно
            with frame_lock:
                current_analysis = analysis
                analysis_timestamp = time.time()
            
            # Визуализация pinch линии (без текста)
            tips = tracker.get_finger_tips(landmarks)
            if 'thumb' in tips and 'index' in tips:
                pt1 = (int(tips['thumb'][0]), int(tips['thumb'][1]))
                pt2 = (int(tips['index'][0]), int(tips['index'][1]))
                color = PickGripClassifier.COLORS[analysis.grip_type]
                cv2.line(frame, pt1, pt2, color, 3)
                cv2.circle(frame, pt1, 8, color, -1)
                cv2.circle(frame, pt2, 8, color, -1)
        
        # Кодируем в JPEG с пониженным качеством для скорости
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Главная страница."""
    return render_template('index.html', tolerance=tolerance)


@app.route('/video_feed')
def video_feed():
    """MJPEG стриминг видео."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    })


@app.route('/api/analysis')
def get_analysis():
    """API endpoint для получения данных анализа."""
    global current_analysis, analysis_timestamp
    
    if current_analysis is None:
        return jsonify({
            'status': 'error',
            'grip_type': 'Кисть не обнаружена',
            'confidence': 0,
            'pinch_distance': 0,
            'hand_rotation': 0,
            'recommendation': 'Поместите руку в кадр камеры',
            'timestamp': 0,
            'age_ms': 0
        })
    
    analysis = current_analysis
    
    # Определяем CSS класс в зависимости от типа хвата
    status_class = 'correct'
    if analysis.grip_type == GripType.CORRECT:
        status_class = 'correct'
    elif analysis.grip_type == GripType.TOO_TIGHT:
        status_class = 'error'
    elif analysis.grip_type == GripType.FINGERS_OPEN:
        status_class = 'warning'
    elif analysis.grip_type == GripType.ROTATED:
        status_class = 'error'
    
    # Вычисляем задержку анализа
    age = time.time() - analysis_timestamp
    
    return jsonify({
        'status': status_class,
        'grip_type': analysis.grip_type.value,
        'confidence': round(analysis.confidence * 100, 1),
        'pinch_distance': round(analysis.pinch_distance, 1),
        'hand_rotation': round(analysis.hand_rotation, 1),
        'recommendation': analysis.recommendation,
        'timestamp': analysis_timestamp,
        'age_ms': round(age * 1000, 0)
    })


@app.route('/api/tolerance', methods=['POST'])
def set_tolerance():
    """Установка порога допустимого отклонения."""
    global tolerance, classifier
    
    data = request.get_json()
    if 'tolerance' not in data:
        return jsonify({'error': 'Tolerance value required'}), 400
    
    new_tolerance = float(data['tolerance'])
    
    # Ограничиваем диапазон 0.05 - 0.50
    new_tolerance = max(0.05, min(0.50, new_tolerance))
    tolerance = new_tolerance
    
    # Обновляем классификатор
    if classifier:
        classifier.tolerance = new_tolerance
    
    return jsonify({
        'status': 'success',
        'tolerance': tolerance,
        'message': f'Tolerance set to {tolerance:.0%}'
    })


@app.route('/api/reference', methods=['POST'])
def upload_reference():
    """Загрузка эталонного изображения."""
    from werkzeug.utils import secure_filename
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Загружаем как эталон
        success = classifier.load_reference_image(filepath)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Reference image loaded successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load reference image'
            }), 400
    
    return jsonify({'error': 'Upload failed'}), 400


def cleanup():
    """Очистка ресурсов."""
    global is_running, camera
    
    is_running = False
    
    if camera:
        camera.release()
    
    if tracker:
        tracker.close()
    
    if classifier:
        classifier.close()


if __name__ == '__main__':
    reference_image = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not init_components(reference_image):
        print("Failed to initialize components")
        sys.exit(1)
    
    print("=" * 60)
    print("GuitarVision Web Service")
    print("=" * 60)
    
    if reference_image:
        print(f"Reference image: {reference_image}")
    else:
        print("Using default grip thresholds")
    
    print(f"Tolerance: {tolerance:.0%}")
    print("\nOpen in browser: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cleanup()
