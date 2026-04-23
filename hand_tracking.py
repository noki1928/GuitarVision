"""
Отслеживание точек кисти в реальном времени через Mediapipe
с классификацией хвата медиатора.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import sys
from grip_classifier import PickGripClassifier, GripType
from audio_feedback import AudioFeedback


class HandTracker:
    """Класс для отслеживания кисти руки."""

    def __init__(self, mode: bool = False, max_hands: int = 1, detection_conf: float = 0.7, tracking_conf: float = 0.5):
        """
        Инициализация трекера кисти.

        Args:
            mode: Если True, каждый кадр обрабатывается как новый (для статичных изображений)
            max_hands: Максимальное количество отслеживаемых рук
            detection_conf: Порог уверенности детекции (0-1)
            tracking_conf: Порог уверенности трекинга (0-1)
        """
        # Путь к модели
        model_path = os.path.join(os.path.dirname(__file__), 'hands.task')
        base_options = python.BaseOptions(model_asset_path=model_path)

        # Используем режим IMAGE для синхронной работы
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.cached_landmarks = None  # Кэш для избежания повторной детекции

    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Обнаружение рук на кадре.

        Args:
            frame: Кадр изображения (BGR)
            draw: Рисовать ли аннотации

        Returns:
            Кадр с аннотациями (если draw=True)
        """
        # Конвертируем в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Создаём Image для Mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Синхронная детекция
        self.results = self.detector.detect(mp_image)

        # Кэшируем landmarks для повторного использования
        if self.results and self.results.hand_landmarks:
            hand_lms = self.results.hand_landmarks[0]
            h, w, _ = frame.shape
            self.cached_landmarks = []
            for lm in hand_lms:
                x, y, z = lm.x * w, lm.y * h, lm.z
                self.cached_landmarks.append(np.array([x, y, z]))
        else:
            self.cached_landmarks = None

        # Рисуем landmarks
        if self.results and self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                # Конвертируем landmarks в формат для отрисовки
                normalized_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                self._draw_landmarks(frame, normalized_landmarks)

        return frame

    def _draw_landmarks(self, frame: np.ndarray, landmarks: list):
        """Рисование landmarks на кадре."""
        h, w, _ = frame.shape

        # Рисуем соединения (кости)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Большой палец
            (0, 5), (5, 6), (6, 7), (7, 8),  # Указательный
            (0, 9), (9, 10), (10, 11), (11, 12),  # Средний
            (0, 13), (13, 14), (14, 15), (15, 16),  # Безымянный
            (0, 17), (17, 18), (18, 19), (19, 20),  # Мизинец
            (5, 9), (9, 13), (13, 17)  # Костяшки
        ]

        for i, j in connections:
            if i < len(landmarks) and j < len(landmarks):
                pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
                pt2 = (int(landmarks[j][0] * w), int(landmarks[j][1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Рисуем точки
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def get_landmarks(self, frame: np.ndarray, hand_num: int = 0) -> list | None:
        """
        DEPRECATED: Используйте get_cached_landmarks() для избежания повторной детекции.
        
        Получение координат landmarks для указанной руки.

        Args:
            frame: Кадр изображения
            hand_num: Номер руки (0-based)

        Returns:
            Список координат landmarks или None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = self.detector.detect(mp_image)

        if results and results.hand_landmarks:
            if hand_num < len(results.hand_landmarks):
                landmarks = []
                hand_lms = results.hand_landmarks[hand_num]
                h, w, _ = frame.shape

                for lm in hand_lms:
                    x, y, z = lm.x * w, lm.y * h, lm.z
                    landmarks.append(np.array([x, y, z]))
                return landmarks
        return None
    
    def get_cached_landmarks(self) -> list | None:
        """
        Получение кэшированных landmarks из последнего вызова find_hands().
        Избегает повторной детекции MediaPipe.

        Returns:
            Список координат landmarks или None
        """
        return self.cached_landmarks

    def get_finger_tips(self, landmarks: list) -> dict:
        """
        Получение координат кончиков пальцев.

        Args:
            landmarks: Список координат landmarks

        Returns:
            Словарь с координатами кончиков пальцев
        """
        if not landmarks:
            return {}

        # Индексы кончиков пальцев в Mediapipe
        finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }

        return {
            name: landmarks[idx] for name, idx in finger_tips.items()
        }

    def is_finger_extended(self, landmarks: list, finger_idx: int) -> bool:
        """
        Проверка, выпрямлен ли палец.

        Args:
            landmarks: Список координат landmarks
            finger_idx: Индекс пальца (4=большой, 8=указательный, 12=средний, 16=безымянный, 20=мизинец)

        Returns:
            True если палец выпрямлен
        """
        if not landmarks or len(landmarks) < finger_idx + 1:
            return False

        # Для большого пальца особая логика
        if finger_idx == 4:
            tip_x = landmarks[4][0]
            base_x = landmarks[2][0]
            return abs(tip_x - base_x) > 30

        # Для остальных пальцев: сравниваем Y-координаты
        tip_y = landmarks[finger_idx][1]
        pip_y = landmarks[finger_idx - 1][1]
        mcp_y = landmarks[finger_idx - 2][1]

        return tip_y < pip_y and tip_y < mcp_y

    def get_extended_fingers(self, landmarks: list) -> list:
        """
        Получение списка выпрямленных пальцев.

        Args:
            landmarks: Список координат landmarks

        Returns:
            Список названий выпрямленных пальцев
        """
        if not landmarks:
            return []

        finger_names = {
            4: 'thumb',
            8: 'index',
            12: 'middle',
            16: 'ring',
            20: 'pinky'
        }

        extended = []
        for idx, name in finger_names.items():
            if self.is_finger_extended(landmarks, idx):
                extended.append(name)
        return extended

    def close(self):
        """Закрытие детектора."""
        if self.detector:
            self.detector.close()


def calculate_distance(point1: tuple, point2: tuple) -> float:
    """Вычисление расстояния между двумя точками."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def main():
    """Основная функция для запуска отслеживания кисти с классификацией хвата."""
    # Проверяем аргумент командной строки для эталонного изображения
    reference_image = None
    if len(sys.argv) > 1:
        reference_image = sys.argv[1]
        if not os.path.exists(reference_image):
            print(f"Error: Reference image not found: {reference_image}")
            return
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    tracker = HandTracker()
    
    # Создаём классификатор с эталонным изображением (если указано)
    if reference_image:
        grip_classifier = PickGripClassifier(reference_image_path=reference_image)
    else:
        grip_classifier = PickGripClassifier()
        print("No reference image provided. Using default grip thresholds.")
        print("To use a reference image, run: python hand_tracking.py <path_to_image>")
    
    audio_feedback = AudioFeedback(enabled=True)

    print("Hand tracking with pick grip classification started")
    print("Press 'q' to quit, 's' for screenshot, 'a' to toggle audio")

    frame_count = 0
    last_landmarks = None
    last_analysis = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame_count += 1
            
            # Зеркальное отражение (flip) для естественного отображения
            frame = cv2.flip(frame, 1)

            # Отслеживание кисти (рисуем всегда)
            frame = tracker.find_hands(frame, draw=True)

            # Получаем landmarks для анализа
            landmarks = tracker.get_landmarks(frame)

            if landmarks:
                last_landmarks = landmarks
                
                # Анализируем хват классификатором
                analysis = grip_classifier.analyze_grip(landmarks)
                last_analysis = analysis
                
                # Звуковая обратная связь при изменении хвата
                audio_feedback.on_grip_change(analysis.grip_type)
                
                # Отрисовка результатов анализа
                grip_classifier.draw_analysis(frame, analysis)
                
                # Получение кончиков пальцев для дополнительной визуализации
                tips = tracker.get_finger_tips(landmarks)
                
                # Визуализация линии между большим и указательным пальцем
                if 'thumb' in tips and 'index' in tips:
                    pt1 = (int(tips['thumb'][0]), int(tips['thumb'][1]))
                    pt2 = (int(tips['index'][0]), int(tips['index'][1]))
                    color = PickGripClassifier.COLORS[analysis.grip_type]
                    cv2.line(frame, pt1, pt2, color, 3)
                    
                    # Рисуем точки на кончиках
                    cv2.circle(frame, pt1, 8, color, -1)
                    cv2.circle(frame, pt2, 8, color, -1)
                    
            elif last_landmarks and last_analysis:
                # Показываем последние известные данные
                grip_classifier.draw_analysis(frame, last_analysis)
                
                tips = tracker.get_finger_tips(last_landmarks)
                if 'thumb' in tips and 'index' in tips:
                    pt1 = (int(tips['thumb'][0]), int(tips['thumb'][1]))
                    pt2 = (int(tips['index'][0]), int(tips['index'][1]))
                    color = PickGripClassifier.COLORS[last_analysis.grip_type]
                    cv2.line(frame, pt1, pt2, color, 3)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

            # Статус звука
            audio_status = "Audio: ON" if audio_feedback.enabled else "Audio: OFF"
            audio_color = (0, 255, 0) if audio_feedback.enabled else (0, 0, 255)
            h, w, _ = frame.shape
            cv2.putText(frame, audio_status, (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)

            # Отображение кадра
            cv2.imshow('GuitarVision - Pick Grip Analysis', frame)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('screenshot.png', frame)
                print("Screenshot saved as screenshot.png")
            elif key == ord('a'):
                state = audio_feedback.toggle()
                print(f"Audio feedback: {'ON' if state else 'OFF'}")

    finally:
        tracker.close()
        grip_classifier.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == "__main__":
    main()
