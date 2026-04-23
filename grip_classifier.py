"""
Классификатор хвата медиатора на основе геометрии кисти.
Определяет правильность хвата и даёт рекомендации по исправлению.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time
import os

class GripType(Enum):
    """Типы хвата (упрощенная классификация)."""
    CORRECT = "Правильно"
    FINGERS_OPEN = "Пальцы разжаты"
    TOO_TIGHT = "Пальцы сильно сжаты"
    ROTATED = "Кисть повернута"


@dataclass
class GripAnalysis:
    """Результат анализа хвата."""
    grip_type: GripType
    confidence: float
    pinch_distance: float
    hand_rotation: float  # Угол поворота кисти
    finger_positions: Dict[str, bool]
    recommendation: str = ""
    timestamp: float = field(default_factory=time.time)


class PickGripClassifier:
    """
    Классификатор хвата медиатора на основе геометрических правил.

    Анализирует положение пальцев и определяет качество хвата.
    """

    # Цвета для разных типов хвата
    COLORS: Dict[GripType, Tuple[int, int, int]] = {
        GripType.CORRECT: (0, 255, 0),        # Зелёный
        GripType.FINGERS_OPEN: (0, 165, 255), # Оранжевый
        GripType.TOO_TIGHT: (0, 0, 255),      # Красный
        GripType.ROTATED: (255, 0, 255),      # Фиолетовый
    }

    # Рекомендации для каждого типа хвата
    RECOMMENDATIONS: Dict[GripType, str] = {
        GripType.CORRECT: "Отлично! Продолжайте держать так",
        GripType.FINGERS_OPEN: "Сожмите пальцы! Большой и указательный должны быть ближе",
        GripType.TOO_TIGHT: "Ослабьте хват! Пальцы слишком сильно сжаты",
        GripType.ROTATED: "Поверните кисть! Положение отличается от эталона",
    }

    def __init__(self, history_length: int = 30, reference_image_path: str = None, tolerance: float = 0.15):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os
        
        # Используем ту же модель, что и в HandTracker
        model_path = os.path.join(os.path.dirname(__file__), 'hands.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.history_length = history_length
        self.distance_history: List[float] = []
        self.last_analysis: Optional[GripAnalysis] = None
        
        # Порог допустимого отклонения от эталона (0.0 = идеально точно, 1.0 = очень свободно)
        # По умолчанию строгая проверка: 15% отклонения
        self.tolerance = tolerance
        
        # Эталонные метрики (загружаются из фотографии)
        self.reference_metrics: Optional[Dict] = None
        if reference_image_path:
            self.load_reference_image(reference_image_path)
    
    def load_reference_image(self, image_path: str) -> bool:
        """
        Загрузка эталонной фотографии правильного хвата.
        Извлекает метрики из изображения и использует их как эталон.
        
        Args:
            image_path: Путь к фотографии эталонного хвата
            
        Returns:
            True если успешно загружено
        """
        if not os.path.exists(image_path):
            print(f"Reference image not found: {image_path}")
            return False
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load reference image: {image_path}")
            return False
        
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            print("No hand detected in reference image")
            return False
        
        # Извлекаем метрики из эталонного изображения
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]
        
        # 1. Размер ладони (для нормализации)
        hand_size = self.calculate_distance(wrist, middle_mcp)
        
        # 2. Нормализованное расстояние щипка
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        normalized_pinch = pinch_distance / hand_size
        
        # 3. Сгиб пальцев (средний, безымянный, мизинец)
        middle_curl = self.calculate_distance(middle_tip, middle_mcp) / hand_size
        ring_curl = self.calculate_distance(ring_tip, ring_mcp) / hand_size
        pinky_curl = self.calculate_distance(pinky_tip, pinky_mcp) / hand_size
        avg_curl = (middle_curl + ring_curl + pinky_curl) / 3
        
        # 4. Поворот кисти (вектор запястье → указательный MCP)
        rotation_vector = index_mcp - wrist
        rotation_angle = np.degrees(np.arctan2(rotation_vector[1], rotation_vector[0]))
        
        # Ключевые метрики эталона
        self.reference_metrics = {
            'hand_size': float(hand_size),
            'normalized_pinch': float(normalized_pinch),
            'avg_curl': float(avg_curl),
            'rotation_angle': float(rotation_angle),
            'rotation_vector': rotation_vector[:2].copy(),  # Сохраняем вектор для точного сравнения
        }
        
        print(f"Reference grip loaded successfully:")
        print(f"  Hand size: {hand_size:.1f}")
        print(f"  Normalized pinch: {normalized_pinch:.3f}")
        print(f"  Avg finger curl: {avg_curl:.3f}")
        print(f"  Rotation angle: {rotation_angle:.1f}°")
        
        return True

    def get_landmarks(self, frame: np.ndarray) -> Optional[list]:
        """Получение landmarks кисти."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.detector.detect(mp_image)

        if results and results.hand_landmarks:
            landmarks = []
            hand_lms = results.hand_landmarks[0]
            h, w, _ = frame.shape

            for lm in hand_lms:
                x, y, z = lm.x * w, lm.y * h, lm.z
                landmarks.append(np.array([x, y, z]))
            return landmarks
        return None

    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Расстояние между двумя точками."""
        return np.linalg.norm(p1[:2] - p2[:2])

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Угол между векторами p1->p2 и p2->p3.

        Args:
            p1: Первая точка
            p2: Вершина угла
            p3: Третья точка

        Returns:
            Угол в градусах
        """
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1[:2], v2[:2]) / (np.linalg.norm(v1[:2]) * np.linalg.norm(v2[:2]))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def calculate_hand_rotation(self, landmarks: list) -> float:
        """
        Вычисление угла поворота кисти.
        Использует вектор от запястья до среднего пальца MCP.
        
        Args:
            landmarks: Список landmarks
            
        Returns:
            Угол поворота в градусах (-180 до 180)
        """
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Вектор от запястья к среднему пальцу
        vector = middle_mcp[:2] - wrist[:2]
        
        # Угол относительно горизонтали
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        
        return angle

    def is_finger_extended(self, landmarks: list, finger_idx: int) -> bool:
        """Проверка, выпрямлен ли палец."""
        if not landmarks or len(landmarks) < finger_idx + 1:
            return False

        if finger_idx == 4:  # Большой палец
            tip_x = landmarks[4][0]
            base_x = landmarks[2][0]
            return abs(tip_x - base_x) > 30

        tip_y = landmarks[finger_idx][1]
        pip_y = landmarks[finger_idx - 1][1]
        mcp_y = landmarks[finger_idx - 2][1]

        return tip_y < pip_y and tip_y < mcp_y

    def calculate_finger_curvature(self, landmarks: list, finger_tip_idx: int) -> float:
        """
        Расчёт кривизны пальца (насколько он согнут).
        
        Args:
            landmarks: Список landmarks
            finger_tip_idx: Индекс кончика пальца
            
        Returns:
            Угол сгиба пальца в градусах (больше = более согнут)
        """
        if finger_tip_idx == 4:  # Большой палец
            return self.calculate_angle(landmarks[2], landmarks[3], landmarks[4])
        else:
            # Для остальных пальцев: угол между MCP-PIP-DIP
            mcp = landmarks[finger_tip_idx - 3]
            pip = landmarks[finger_tip_idx - 2]
            dip = landmarks[finger_tip_idx - 1]
            tip = landmarks[finger_tip_idx]
            
            # Средний угол сгиба
            angle1 = self.calculate_angle(mcp, pip, dip)
            angle2 = self.calculate_angle(pip, dip, tip)
            return (angle1 + angle2) / 2

    def calculate_thumb_position_score(self, landmarks: list) -> float:
        """
        Оценка положения большого пальца относительно указательного.
        
        Returns:
            0.0 = идеальное положение, >1.0 = плохое положение
        """
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        
        # Расстояние от большого пальца до указательного
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        # Расстояние от MCP большого пальца до ладони
        wrist = landmarks[0]
        thumb_mcp_dist = self.calculate_distance(thumb_mcp, wrist)
        
        # Нормализованная оценка
        ratio = thumb_index_dist / max(thumb_mcp_dist, 1.0)
        return ratio

    def calculate_hand_tension(self, landmarks: list) -> float:
        """
        Оценка напряжения кисти.
        
        Returns:
            0.0 = расслаблена, >1.0 = напряжена
        """
        # Измеряем расстояния между кончиками пальцев
        tips = [8, 12, 16, 20]  # указательный, средний, безымянный, мизинец
        distances = []
        
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                dist = self.calculate_distance(landmarks[tips[i]], landmarks[tips[j]])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Нормализация (эмпирические значения)
        tension = max(0, (avg_distance - 30) / 50)
        return tension

    def analyze_grip(self, landmarks: list) -> GripAnalysis:
        """
        Анализ хвата с нормализацией и улучшенной классификацией.

        Args:
            landmarks: Список из 21 landmarks кисти

        Returns:
            GripAnalysis с результатами
        """
        if not landmarks:
            return GripAnalysis(
                grip_type=GripType.FINGERS_OPEN,
                confidence=0.0,
                pinch_distance=0.0,
                hand_rotation=0.0,
                finger_positions={},
                recommendation=self.RECOMMENDATIONS[GripType.FINGERS_OPEN]
            )

        # Ключевые точки
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]

        # 1. Размер ладони (для нормализации)
        hand_size = self.calculate_distance(wrist, middle_mcp)
        
        # 2. Нормализованное расстояние щипка
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        normalized_pinch = pinch_distance / hand_size
        
        # 3. Сгиб пальцев (средний, безымянный, мизинец)
        middle_curl = self.calculate_distance(middle_tip, middle_mcp) / hand_size
        ring_curl = self.calculate_distance(ring_tip, ring_mcp) / hand_size
        pinky_curl = self.calculate_distance(pinky_tip, pinky_mcp) / hand_size
        avg_curl = (middle_curl + ring_curl + pinky_curl) / 3
        
        # 4. Поворот кисти (вектор запястье → указательный MCP)
        rotation_vector = index_mcp - wrist
        rotation_angle = np.degrees(np.arctan2(rotation_vector[1], rotation_vector[0]))

        # Положение пальцев (для дополнительной проверки)
        finger_positions = {
            'thumb_extended': self.is_finger_extended(landmarks, 4),
            'index_extended': self.is_finger_extended(landmarks, 8),
        }

        # Классификация с новыми метриками
        grip_type, confidence = self._classify(
            normalized_pinch, avg_curl, rotation_angle, rotation_vector, finger_positions
        )

        analysis = GripAnalysis(
            grip_type=grip_type,
            confidence=confidence,
            pinch_distance=pinch_distance,
            hand_rotation=rotation_angle,
            finger_positions=finger_positions,
            recommendation=self.RECOMMENDATIONS[grip_type]
        )
        
        self.last_analysis = analysis
        return analysis

    def _classify(self, normalized_pinch: float, avg_curl: float, 
                  rotation_angle: float, rotation_vector: np.ndarray,
                  finger_positions: Dict[str, bool]
                  ) -> Tuple[GripType, float]:
        """
        Улучшенная классификация хвата (4 типа) с нормализацией.
        
        Приоритет проверок:
        1. Угол поворота кисти (наиболее важно)
        2. Сгиб пальцев (средний, безымянный, мизинец)
        3. Расстояние щипка (большой-указательный)
        
        Требует наличия эталонного изображения.
        """
        # Эталон обязателен
        if not self.reference_metrics:
            return GripType.FINGERS_OPEN, 0.5
        
        ref = self.reference_metrics
        
        # Пороги для тестирования
        rotation_threshold = 30  # 40° для поворота кисти
        curl_tolerance_tight = 0.50  # 50% для TOO_TIGHT (сжатые пальцы)
        curl_tolerance_open = 1.50   # 150% для FINGERS_OPEN (разжатые пальцы - только когда совсем прямые)
        pinch_tolerance = self.tolerance  # 15% по умолчанию
        
        # ПРИОРИТЕТ 1: Проверка угла поворота кисти (самое важное)
        # Вычисляем угол между векторами
        ref_vector = ref['rotation_vector']
        current_vector = rotation_vector[:2]
        
        # Угол между векторами через скалярное произведение
        cos_angle = np.dot(ref_vector, current_vector) / (
            np.linalg.norm(ref_vector) * np.linalg.norm(current_vector)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rotation_deviation = np.degrees(np.arccos(cos_angle))
        
        if rotation_deviation > rotation_threshold:
            confidence = min(1.0, rotation_deviation / 90)
            return GripType.ROTATED, confidence
        
        # ПРИОРИТЕТ 2: Проверка сгиба пальцев (средний, безымянный, мизинец)
        # Создаем асимметричное "окно правильности" - разжатость детектируется реже
        curl_lower_bound = ref['avg_curl'] * (1 - curl_tolerance_tight)
        curl_upper_bound = ref['avg_curl'] * (1 + curl_tolerance_open)
        
        # Пальцы слишком сжаты (меньше нижней границы)
        if avg_curl < curl_lower_bound:
            curl_deviation = (curl_lower_bound - avg_curl) / ref['avg_curl']
            confidence = min(1.0, curl_deviation + 0.5)
            return GripType.TOO_TIGHT, confidence
        
        # Пальцы слишком разжаты (больше верхней границы - только когда совсем прямые)
        if avg_curl > curl_upper_bound:
            curl_deviation = (avg_curl - curl_upper_bound) / ref['avg_curl']
            confidence = min(1.0, curl_deviation + 0.5)
            return GripType.FINGERS_OPEN, confidence
        
        # Сгиб пальцев в пределах нормы - хват правильный!
        # Расстояние щипка больше не проверяется
        return GripType.CORRECT, 0.9

    def draw_analysis(self, frame: np.ndarray, analysis: GripAnalysis) -> None:
        """
        Отрисовка результатов анализа на кадре.

        Args:
            frame: Кадр для отрисовки
            analysis: Результаты анализа
        """
        color = self.COLORS[analysis.grip_type]

        # Основной текст с типом хвата
        grip_text = f"Grip: {analysis.grip_type.value}"
        cv2.putText(frame, grip_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        # Детали
        details = [
            f"Pinch dist: {analysis.pinch_distance:.1f}",
            f"Pinch angle: {analysis.pinch_angle:.1f}°",
            f"Thumb angle: {analysis.thumb_angle:.1f}°",
            f"Confidence: {analysis.confidence:.0%}"
        ]

        for i, detail in enumerate(details):
            y_pos = 60 + i * 25
            cv2.putText(frame, detail, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Рекомендация
        if analysis.recommendation:
            rec_lines = analysis.recommendation.split('\n')
            for i, line in enumerate(rec_lines):
                y_pos = 160 + i * 25
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Индикатор правильности (полоска сбоку)
        h, w, _ = frame.shape
        bar_width = int(w * 0.03)
        bar_height = int(h * analysis.confidence)

        cv2.rectangle(frame, (w - bar_width, h - bar_height),
                     (w, h), color, -1)
        cv2.rectangle(frame, (w - bar_width, 0),
                     (w, h), (255, 255, 255), 2)
        
        # Визуализация кривизны пальцев
        self._draw_finger_curvature(frame, analysis)

    def _draw_finger_curvature(self, frame: np.ndarray, analysis: GripAnalysis) -> None:
        """Отрисовка индикаторов кривизны пальцев."""
        if not analysis.finger_curvature:
            return
            
        h, w, _ = frame.shape
        start_x = 10
        start_y = h - 120
        
        finger_names_ru = {
            'thumb': 'Thumb',
            'index': 'Index',
            'middle': 'Middle',
            'ring': 'Ring',
            'pinky': 'Pinky'
        }
        
        for i, (finger, curvature) in enumerate(analysis.finger_curvature.items()):
            y = start_y + i * 20
            
            # Цвет в зависимости от кривизны
            if curvature < 30:
                color = (0, 0, 255)  # Прямой - красный
            elif curvature < 60:
                color = (0, 165, 255)  # Средний - оранжевый
            else:
                color = (0, 255, 0)  # Согнут - зелёный
            
            # Полоска
            bar_width = int(min(curvature / 180.0, 1.0) * 50)
            cv2.rectangle(frame, (start_x + 80, y - 10),
                         (start_x + 80 + bar_width, y + 5), color, -1)
            
            # Текст
            cv2.putText(frame, finger_names_ru.get(finger, finger),
                       (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_visual_feedback(self, frame: np.ndarray, analysis: GripAnalysis) -> None:
        """Рисование визуальных индикаторов на landmarks."""
        pass  # Можно добавить визуализацию позже

    def close(self):
        """Закрытие ресурса."""
        self.detector.close()


def main():
    """Запуск классификатора хвата."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    classifier = PickGripClassifier()
    print("Классификатор хвата медиатора запущен")
    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получение landmarks
        landmarks = classifier.get_landmarks(frame)

        if landmarks:
            # Анализ хвата
            analysis = classifier.analyze_grip(landmarks)
            # Отрисовка
            classifier.draw_analysis(frame, analysis)
        else:
            cv2.putText(frame, "Рука не обнаружена", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

        cv2.imshow('Pick Grip Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
