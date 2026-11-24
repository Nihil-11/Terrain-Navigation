import numpy as np
from config import NUM_BEAMS, LIDAR_TILT_RAD, BEAM_TILTS, BEAM_AZIMUTHS

# Заранее считаем синусы и косинусы наклонов для скорости
_SIN_TILTS = np.sin(BEAM_TILTS)
_COS_TILTS = np.cos(BEAM_TILTS)

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
    angles = yaw + BEAM_AZIMUTHS
    dx = r * np.cos(angles)
    dy = r * np.sin(angles)

    # Собираем результат в матрицу (51, 3)[cite: 6]
    return np.column_stack((dx, dy, dz))