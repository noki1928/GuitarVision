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
    """Pick grip types."""
    CORRECT = "Correct Grip"
    TOO_TIGHT = "Too Tight"
    TOO_LOOSE = "Too Loose"
    WRONG_ANGLE = "Wrong Pinch Angle"
    FINGERS_STRAIGHT = "Fingers Not Curved"
    NO_HAND = "No Hand Detected"
    WRONG_THUMB_POSITION = "Wrong Thumb Position"
    TENSE_HAND = "Hand Tension"


@dataclass
class GripAnalysis:
    """Результат анализа хвата."""
    grip_type: GripType
    confidence: float
    pinch_distance: float
    pinch_angle: float
    thumb_angle: float
    finger_curvature: Dict[str, float]
    finger_positions: Dict[str, bool]
    recommendation: str = ""
    history: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class PickGripClassifier:
    """
    Классификатор хвата медиатора на основе геометрических правил.

    Анализирует положение пальцев и определяет качество хвата.
    """

    # Цвета для разных типов хвата
    COLORS: Dict[GripType, Tuple[int, int, int]] = {
        GripType.CORRECT: (0, 255, 0),        # Зелёный
        GripType.TOO_TIGHT: (0, 0, 255),      # Красный
        GripType.TOO_LOOSE: (0, 165, 255),    # Оранжевый
        GripType.WRONG_ANGLE: (255, 0, 255),  # Фиолетовый
        GripType.FINGERS_STRAIGHT: (255, 255, 0),  # Жёлтый
        GripType.NO_HAND: (128, 128, 128),    # Серый
        GripType.WRONG_THUMB_POSITION: (0, 128, 255),  # Синий-оранжевый
        GripType.TENSE_HAND: (128, 0, 128),   # Тёмно-фиолетовый
    }

    # Рекомендации для каждого типа хвата
    RECOMMENDATIONS: Dict[GripType, str] = {
        GripType.CORRECT: "Great! Keep playing with this grip",
        GripType.TOO_TIGHT: "Loosen your grip! Relax your hand, the pick should move freely",
        GripType.TOO_LOOSE: "Tighten your grip! The pick might slip out",
        GripType.WRONG_ANGLE: "Change the pick angle. Hold it at 15-30° to the strings",
        GripType.FINGERS_STRAIGHT: "Curve your fingers! Middle, ring, and pinky should be relaxed and curved",
        GripType.NO_HAND: "Show your hand to the camera",
        GripType.WRONG_THUMB_POSITION: "Place thumb opposite to index finger, don't extend it too far",
        GripType.TENSE_HAND: "Relax your hand! Tension hinders playing technique",
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
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        
        # Ключевые метрики эталона
        self.reference_metrics = {
            'pinch_distance': float(self.calculate_distance(thumb_tip, index_tip)),
            'pinch_angle': float(self.calculate_angle(thumb_ip, thumb_tip, index_tip)),
            'thumb_angle': float(self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)),
        }
        
        print(f"Reference grip loaded successfully:")
        print(f"  Pinch distance: {self.reference_metrics['pinch_distance']:.1f}")
        print(f"  Pinch angle: {self.reference_metrics['pinch_angle']:.1f}°")
        print(f"  Thumb angle: {self.reference_metrics['thumb_angle']:.1f}°")
        
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
        Анализ хвата медиатора.

        Args:
            landmarks: Список из 21 landmarks кисти

        Returns:
            GripAnalysis с результатами
        """
        if not landmarks:
            return GripAnalysis(
                grip_type=GripType.NO_HAND,
                confidence=0.0,
                pinch_distance=0.0,
                pinch_angle=0.0,
                thumb_angle=0.0,
                finger_curvature={},
                finger_positions={},
                recommendation=self.RECOMMENDATIONS[GripType.NO_HAND]
            )

        # Ключевые точки
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]  # Межфаланговый сустав
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Расстояние pinch (большой-указательный)
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)

        # Угол щипка
        pinch_angle = self.calculate_angle(thumb_ip, thumb_tip, index_tip)
        
        # Угол большого пальца
        thumb_angle = self.calculate_angle(thumb_mcp, thumb_ip, thumb_tip)

        # Кривизна пальцев
        finger_curvature = {
            'thumb': self.calculate_finger_curvature(landmarks, 4),
            'index': self.calculate_finger_curvature(landmarks, 8),
            'middle': self.calculate_finger_curvature(landmarks, 12),
            'ring': self.calculate_finger_curvature(landmarks, 16),
            'pinky': self.calculate_finger_curvature(landmarks, 20),
        }

        # Положение пальцев
        finger_positions = {
            'thumb_extended': self.is_finger_extended(landmarks, 4),
            'index_extended': self.is_finger_extended(landmarks, 8),
            'middle_flexed': not self.is_finger_extended(landmarks, 12),
            'ring_flexed': not self.is_finger_extended(landmarks, 16),
            'pinky_flexed': not self.is_finger_extended(landmarks, 20),
        }

        # Дополнительные метрики
        thumb_position_score = self.calculate_thumb_position_score(landmarks)
        hand_tension = self.calculate_hand_tension(landmarks)

        # Добавляем в историю
        self.distance_history.append(pinch_distance)
        if len(self.distance_history) > self.history_length:
            self.distance_history.pop(0)

        # Классификация
        grip_type, confidence = self._classify(
            pinch_distance, pinch_angle, thumb_angle,
            finger_positions, finger_curvature,
            thumb_position_score, hand_tension
        )

        analysis = GripAnalysis(
            grip_type=grip_type,
            confidence=confidence,
            pinch_distance=pinch_distance,
            pinch_angle=pinch_angle,
            thumb_angle=thumb_angle,
            finger_curvature=finger_curvature,
            finger_positions=finger_positions,
            recommendation=self.RECOMMENDATIONS[grip_type]
        )
        
        self.last_analysis = analysis
        return analysis

    def _classify(self, pinch_distance: float, pinch_angle: float,
                  thumb_angle: float, finger_positions: Dict[str, bool],
                  finger_curvature: Dict[str, float],
                  thumb_position_score: float, hand_tension: float
                  ) -> Tuple[GripType, float]:
        """
        Классификация хвата на основе правил.
        
        Если загружены эталонные метрики, использует их для сравнения.
        Иначе использует стандартные пороги.

        Пороги настроены эмпирически и могут требовать калибровки.
        """
        # Проверка: все пальцы выпрямлены (нет хвата) - только это считаем неправильным
        if (finger_positions['thumb_extended'] and
            finger_positions['index_extended'] and
            not finger_positions['middle_flexed']):
            return GripType.FINGERS_STRAIGHT, 0.9

        # Если есть эталонные метрики, используем их для классификации
        if self.reference_metrics:
            return self._classify_with_reference(
                pinch_distance, pinch_angle, thumb_angle,
                finger_positions, finger_curvature
            )
        
        # Стандартная классификация (без эталона) - очень мягкая
        # Только проверяем что пальцы согнуты и есть какой-то зажим
        fingers_curved = (
            finger_positions['middle_flexed'] or
            finger_positions['ring_flexed'] or
            finger_positions['pinky_flexed']
        )
        
        if fingers_curved:
            return GripType.CORRECT, 0.8
        
        # Пограничные случаи
        return GripType.CORRECT, 0.5
    
    def _classify_with_reference(self, pinch_distance: float, pinch_angle: float,
                                  thumb_angle: float, finger_positions: Dict[str, bool],
                                  finger_curvature: Dict[str, float]
                                  ) -> Tuple[GripType, float]:
        """
        Классификация на основе сравнения с эталонным изображением.
        Сравниваем ТОЛЬКО расстояние pinch. Угол не учитываем.
        
        self.tolerance определяет допустимое отклонение:
        - 0.05 = очень строго (5% отклонения)
        - 0.15 = строго (15% отклонения, по умолчанию)
        - 0.30 = средне (30% отклонения)
        - 0.50 = мягко (50% отклонения)
        """
        ref = self.reference_metrics
        
        # Вычисляем отклонение ТОЛЬКО по расстоянию
        dist_deviation = abs(pinch_distance - ref['pinch_distance']) / max(ref['pinch_distance'], 1.0)
        
        # Проверяем, НЕ раскрыты ли все пальцы (самая важная проверка)
        all_fingers_straight = (
            finger_positions['thumb_extended'] and
            finger_positions['index_extended'] and
            not finger_positions['middle_flexed']
        )
        
        if all_fingers_straight:
            return GripType.FINGERS_STRAIGHT, 0.9
        
        # Строгая классификация на основе tolerance
        # Если отклонение в пределах допуска - правильный хват
        if dist_deviation <= self.tolerance:
            confidence = max(0.85, 1.0 - dist_deviation)
            return GripType.CORRECT, confidence
        
        # Отклонение超出 tolerance - определяем тип проблемы
        # Слишком сильный зажим (расстояние значительно меньше эталона)
        if pinch_distance < ref['pinch_distance'] * (1 - self.tolerance):
            confidence = min(1.0, (ref['pinch_distance'] - pinch_distance) / ref['pinch_distance'] + 0.5)
            return GripType.TOO_TIGHT, confidence
        
        # Слишком слабый зажим (расстояние значительно больше эталона)
        if pinch_distance > ref['pinch_distance'] * (1 + self.tolerance):
            confidence = min(1.0, (pinch_distance - ref['pinch_distance']) / ref['pinch_distance'] + 0.5)
            return GripType.TOO_LOOSE, confidence
        
        # Небольшое отклонение - всё ещё приемлемо, но не идеально
        if dist_deviation <= self.tolerance * 1.5:
            confidence = max(0.6, 1.0 - dist_deviation)
            return GripType.CORRECT, confidence
        
        # Значительное отклонение - неправильный хват
        if pinch_distance < ref['pinch_distance']:
            return GripType.TOO_TIGHT, min(1.0, dist_deviation)
        else:
            return GripType.TOO_LOOSE, min(1.0, dist_deviation)

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
