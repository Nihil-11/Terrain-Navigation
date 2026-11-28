import numpy as np
"""
TERRAIN_FILE = "terrain.npy"                   # Файл с готовой матрицей высот
MAP_SIZE = 2500                                # Размер матрицы в ячейках (2500 x 2500)
MAP_RESOLUTION = 1.63                          # Масштаб: 1 ячейка = 1.63 метра
MAP_SIZE_METERS = MAP_SIZE * MAP_RESOLUTION    # Физический размер карты в метрах (4075.0 м)
"""

MAPS = {
    "mountains": {"file": "terrain.npy",       "resolution": 1.63},
    "city":      {"file": "terrain_city.npy",  "resolution": 3.58},  
    "plain":     {"file": "terrain_plain.npy", "resolution": 3.25}, 
}

ACTIVE_MAP = "plain"                         #"mountains", "city" или "plain"

TERRAIN_FILE = MAPS[ACTIVE_MAP]["file"]     
MAP_SIZE = 2500                                # Размер матрицы в ячейках (2500 x 2500)
MAP_RESOLUTION = MAPS[ACTIVE_MAP]["resolution"]
MAP_SIZE_METERS = MAP_SIZE * MAP_RESOLUTION    # Физический размер карты в метрах

DT = 0.1                                       # Шаг времени симуляции в секундах
NUM_STEPS = 700                                # Длина симуляции в шагах (50 секунд полета)

NUM_BEAMS = 51
LIDAR_MAX_RANGE = 700.0                        # Предел дальности лидара в метрах
LIDAR_TILT_RAD = np.deg2rad(20.0)              # Угол наклона боковых лучей (20 град в радианах)
FIXED_ALTITUDE_Z = 550.0                       # Фиксированная высота дрона относительно абс. нуля
DRONE_FIXED_SPEED = 20.0                       # Фиксированная скорость дрона м/c
MARGIN = 1601.0                                # Оступ от краев карты для безопасного спавна дрона и частиц
ERROR_LEN = -1.0                               # Возвращает это значение если луч лидара не достает до земли

# Массив углов наклона от вертикали (длина 51):
# индекс 0 -> 0.0 (строго вниз), индексы 1..50 -> 20 градусов в радианах
BEAM_TILTS = np.full(NUM_BEAMS, LIDAR_TILT_RAD, dtype=np.float64)
BEAM_TILTS[0] = 0.0

# Массив азимутов вокруг дрона (длина 51):
# индекс 0 0.0, индексы 1..50 равномерно по кругу от 0 до 2*pi без замыкания
BEAM_AZIMUTHS = np.zeros(NUM_BEAMS, dtype=np.float64)
BEAM_AZIMUTHS[1:] = np.linspace(0.0, 2.0 * np.pi, NUM_BEAMS - 1, endpoint=False)


LIDAR_NOISE_STD = 0.3                          # Погрешность дальномера лидара (м)
GYRO_NOISE_STD = 0.01                          # Погрешность гироскопа (рад/с)
ACCEL_NOISE_STD = 0.05                         # Погрешность сдвига за один шаг dt (м)

# Готовые наборы шумов для передачи в симулятор (рабочий и тестовый без шума)
DEFAULT_NOISE_CFG = {
    "lidar_std": LIDAR_NOISE_STD,
    "gyro_std": GYRO_NOISE_STD,
    "accel_std": ACCEL_NOISE_STD,
}

ZERO_NOISE_CFG = {
    "lidar_std": 0.0,
    "gyro_std": 0.0,
    "accel_std": 0.0,
}   


NUM_PARTICLES = 8000                           # Число гипотез в облаке