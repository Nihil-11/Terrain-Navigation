import numpy as np
from config import NUM_BEAMS, LIDAR_TILT_RAD

# Предвычисляем геометрию 51 луча (выполняется 1 раз при импорте модуля)
# Индекс 0: наклон 0.0 (строго вниз); индексы 1..50: наклон LIDAR_TILT_RAD[cite: 5]
_TILTS = np.zeros(NUM_BEAMS, dtype=np.float64)
_TILTS[1:] = LIDAR_TILT_RAD

# Азимуты: индекс 0 -> 0.0, индексы 1..50 -> равномерно с шагом 2*pi/50 от носа против часовой[cite: 5]
_AZIMUTHS = np.zeros(NUM_BEAMS, dtype=np.float64)
_AZIMUTHS[1:] = np.linspace(0.0, 2.0 * np.pi, NUM_BEAMS - 1, endpoint=False)

# Заранее считаем синусы и косинусы наклонов для скорости
_SIN_TILTS = np.sin(_TILTS)
_COS_TILTS = np.cos(_TILTS)


def get_lidar_offsets(distances: np.ndarray, yaw: float) -> np.ndarray:
    """
    Переводит измеренные дальности 51 луча лидара в смещения (dx, dy, dz)
    относительно дрона в координатах карты[cite: 6].

    :param distances: массив длин лучей формы (51,) в метрах[cite: 6]
    :param yaw: курс дрона в радианах (0 вдоль +X, против часовой положительный)[cite: 5, 6]
    :return: массив смещений offsets формы (51, 3) в метрах[cite: 6]
    """
    distances = np.asarray(distances, dtype=np.float64)

    # Горизонтальная проекция r и вертикальное смещение dz для каждого луча
    r = distances * _SIN_TILTS
    dz = -distances * _COS_TILTS

    # Поворот в плоскость карты с учетом курса yaw[cite: 9]
    angles = yaw + _AZIMUTHS
    dx = r * np.cos(angles)
    dy = r * np.sin(angles)

    # Собираем результат в матрицу (51, 3)[cite: 6]
    return np.column_stack((dx, dy, dz))