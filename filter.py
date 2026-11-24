import numpy as np
from config import BEAM_TILTS, BEAM_AZIMUTHS, NUM_PARTICLES

# считаем синусы и косинусы наклонов для скорости
_SIN_TILTS = np.sin(BEAM_TILTS)
_COS_TILTS = np.cos(BEAM_TILTS)

# фильтр частиц для оценки состояния дрона по карте высот
class ParticleFilter:
    def __init__(
        self,
        height_map: np.ndarray, # двумерная карта высот
        num_particles: int = NUM_PARTICLES # число частиц/гипотез
    ):
        if num_particles <= 0: raise ValueError("num_particles должно быть положительным числом")

        self.height_map = np.asarray(height_map, dtype=np.float64)
        self.num_particles = int(num_particles)

        self.particles = np.zeros((self.num_particles, 4), dtype=np.float64) # матрица гипотез
        self.weights = np.zeros(self.num_particles, dtype=np.float64) # вектор весов (долей вероятности для частиц)

        self.yaw = 0.0 # начальный курс по умолчанию
    
    

# переводит измеренные дальности 51 луча лидара в смещения (dx, dy, dz)
# относительно дрона в координатах карты
def get_lidar_offsets(distances: np.ndarray, yaw: float) -> np.ndarray:
    # yaw - курс дрона в радианах (0 вдоль +X, против часовой положительный)
    
    # массив длин лучей формы (51, ...) в метрах
    distances = np.asarray(distances, dtype=np.float64)

    # гориз. проекция r и вертикальное смещение dz для каждого луча
    r = distances * _SIN_TILTS
    dz = -distances * _COS_TILTS

    # поворот в плоскость карты с учетом курса yaw
    angles = yaw + BEAM_AZIMUTHS
    dx = r * np.cos(angles)
    dy = r * np.sin(angles)

    # массив смещений offsets формы (51, 3) в метрах (51 трехмерных векторов для каждого луча)
    return np.column_stack((dx, dy, dz))