"""
Модуль звуковой обратной связи для GuitarVision.
Использует системные звуки для оповещения об изменениях хвата.
"""

import threading
import time
from typing import Optional
from grip_classifier import GripType


class AudioFeedback:
    """
    Система звуковой обратной связи.
    
    Воспроизводит звуки при изменении типа хвата,
    чтобы пользователь мог сосредоточиться на игре,
    не глядя постоянно на экран.
    """
    
    # Частоты звуков для разных типов хвата (в Гц)
    FREQUENCIES = {
        GripType.CORRECT: 800,        # Высокий приятный звук
        GripType.TOO_TIGHT: 300,      # Низкий предупреждающий
        GripType.FINGERS_OPEN: 400,   # Средний предупреждающий
        GripType.ROTATED: 350,        # Другой тон
    }
    
    # Длительность звука (мс)
    DURATIONS = {
        GripType.CORRECT: 200,        # Короткий положительный
        GripType.TOO_TIGHT: 400,      # Длинный предупреждающий
        GripType.FINGERS_OPEN: 300,
        GripType.ROTATED: 350,
    }
    
    def __init__(self, enabled: bool = True):
        """
        Инициализация.
        
        Args:
            enabled: Включена ли звуковая обратная связь
        """
        self.enabled = enabled
        self._last_grip_type: Optional[GripType] = None
        self._last_sound_time: float = 0
        self._min_interval: float = 2.0  # Минимальный интервал между звуками (сек)
        self._sound_thread: Optional[threading.Thread] = None
        
    def on_grip_change(self, grip_type: GripType) -> None:
        """
        Вызывается при обнаружении нового типа хвата.
        
        Args:
            grip_type: Текущий тип хвата
        """
        if not self.enabled:
            return
            
        # Не воспроизводим звук слишком часто
        now = time.time()
        if now - self._last_sound_time < self._min_interval:
            return
            
        # Звук только при изменении типа хвата
        if grip_type == self._last_grip_type:
            return
            
        self._last_grip_type = grip_type
        self._last_sound_time = now
        
        # Воспроизводим звук в отдельном потоке
        if self._sound_thread and self._sound_thread.is_alive():
            return  # Не накладываем звуки
            
        self._sound_thread = threading.Thread(
            target=self._play_sound,
            args=(grip_type,),
            daemon=True
        )
        self._sound_thread.start()
        
    def _play_sound(self, grip_type: GripType) -> None:
        """
        Воспроизведение звука через системный динамик.
        
        Args:
            grip_type: Тип хвата для определения тона и длительности
        """
        try:
            import winsound
            
            frequency = self.FREQUENCIES.get(grip_type, 500)
            duration = self.DURATIONS.get(grip_type, 300)
            
            # Для правильного хвата - двойной сигнал
            if grip_type == GripType.CORRECT:
                winsound.Beep(600, 100)
                time.sleep(0.1)
                winsound.Beep(800, 150)
            else:
                winsound.Beep(frequency, duration)
                
        except Exception as e:
            # Если не удалось воспроизвести звук - просто печатаем
            print(f"Звук: {grip_type.value}")
            
    def toggle(self) -> bool:
        """Переключить состояние звуковой обратной связи."""
        self.enabled = not self.enabled
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Установить состояние звуковой обратной связи."""
        self.enabled = enabled
        
    def reset(self) -> None:
        """Сбросить состояние для воспроизведения при следующем изменении."""
        self._last_grip_type = None
