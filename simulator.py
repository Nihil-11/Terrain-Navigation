import numpy as np
from config import (
    TERRAIN_FILE,
    NUM_STEPS,
    MAP_SIZE,
    MAP_RESOLUTION,
    FIXED_ALTITUDE_Z,
    DRONE_FIXED_SPEED,
    MARGIN,
    ERROR_LEN, 
    DT,
    LIDAR_MAX_RANGE,
    LIDAR_TILT_RAD,
    BEAM_TILTS,
    BEAM_AZIMUTHS,  
    DEFAULT_NOISE_CFG,
    ZERO_NOISE_CFG,
)

"""
Словарь основных переменных simulator.py

КАРТА И КООРДИНАТЫ:
- H / height_map: матрица высот местности (2500x2500 ячеек, значения в метрах)
- ix, iy: целочисленные индексы ячейки на карте (перевод метров в пиксели)
- max_x, max_y: физический размер карты в метрах (4075.0 x 4075.0 м)

НАСТРОЙКИ И СОСТОЯНИЕ ДРОНА (внутри DroneSimulator):
- route_name: выбранный сценарий полета ("straight", "snake", "maneuver")
- noise_cfg: словарь со стандартными отклонениями шумов датчиков
- rng: генератор случайных чисел NumPy (управляется параметром seed)
- speed: постоянная скорость полета (м/с)
- margin: минимальный отступ от краев карты при спавне (м)
- min_safe_margin: минимально допустимый отступ (max_flight_dist + max_cone_radius)
- points: число точек, на которые разбивается один луч лидара (600)
- step_count: счетчик пройденных шагов симуляции
- true_x, true_y, true_z: истинные координаты дрона в пространстве (м)
- true_yaw: истинный текущий курс, куда смотрит нос дрона (рад)
- start_yaw: начальный курс при старте (рад), отдается фильтру в момент t = 0

ДВИЖЕНИЕ И ДАТЧИКИ (get_turn_rate, move):
- t: текущее время полета в секундах (step_count * DT)
- turn_rate: угловая скорость поворота носа дрона (рад/с)
- accel_std, gyro_std: уровень шума одометрии (м) и гироскопа (рад/с)
- measured_dist: зашумленный сдвиг вперед за один шаг DT (м)
- measured_dyaw: зашумленный поворот за один шаг DT (рад)

ЛИДАР И ТРАССИРОВКА ЛУЧЕЙ (measure_vertical_lidar, cast_ray, measure_lidar):
- tilt_rad: угол наклона луча от вертикали (рад)
- azimuth_rad: угол луча по кругу относительно носа дрона (рад)
- angle: абсолютный угол направления луча на карте (true_yaw + azimuth_rad, рад)
- R: горизонтальный вынос луча на уровне z = 0 (м)
- end_x, end_y: конечная точка луча на уровне z = 0 (м)
- xs, ys, zs: массивы координат всех точек вдоль луча от дрона до z = 0 (м)
- ground: высота земли из карты под дроном или в точках луча (м)
- hits: массив True/False, где луч опустился ниже или вровень с землей (zs <= ground)
- i: индекс первого True в hits (точка первого касания луча с рельефом)
- true_range: истинное расстояние по прямой от дрона до земли (м)
- lidar_std: уровень шума дальномера лидара (м)
- noisy_range / noise_range: итоговая зашумленная дальность одного луча (м)
- lidar_ranges: массив из 51 замера (0-й строго вниз, 1..50 по конусу, м)
"""

def generate_map(size=MAP_SIZE, seed=10):
    H = np.load(TERRAIN_FILE)
    H = H.astype(np.float64) # распаковываем из float32 в float64
    return H


def get_height(height_map, x, y):
    x_arr = np.asarray(x, dtype=np.float64) # преобразует входные данные x в массив NumPy (numpy.ndarray) с типом данных float64 (числа с плавающей точкой двойной точности) и сохраняет результат в переменную x_arr.
    y_arr = np.asarray(y, dtype=np.float64) 

    ix = np.floor(x_arr / MAP_RESOLUTION).astype(np.int64) # переводит непрерывные координаты (например, в метрах) в дискретные целочисленные индексы ячеек сетки или карты
    iy = np.floor(y_arr / MAP_RESOLUTION).astype(np.int64)

    ix = np.clip(ix, 0, height_map.shape[0] - 1) # ограничивает значения элементов массива заданным интервалом
    iy = np.clip(iy, 0, height_map.shape[1] - 1)        
    
    return height_map[ix, iy]
"""
    ix = int(x // 1.63)
    iy = int(y // 1.63)

    if ix < 0:
        ix = 0
    elif ix > 2499: 
        ix = 2499

    if iy < 0:
        iy = 0
    elif iy > 2499:
        iy = 2499
 """

class DroneSimulator:
    def __init__(self, height_map, route_name="straight", noise_cfg=None, seed=None):
        self.height_map = height_map
        self.route_name = route_name
        self.noise_cfg = DEFAULT_NOISE_CFG if noise_cfg is None else noise_cfg #ai
        self.rng = np.random.default_rng(seed)
        self.speed = DRONE_FIXED_SPEED
        start_z = FIXED_ALTITUDE_Z
        self.margin = MARGIN
        self.points = 600 
        self.step_count = 0

        self.max_x = height_map.shape[0] * MAP_RESOLUTION
        self.max_y = height_map.shape[1] * MAP_RESOLUTION
        self.true_z = float(start_z)

        max_flight_dist = self.speed * NUM_STEPS * DT
        max_cone_radius = self.true_z * np.tan(LIDAR_TILT_RAD)
        min_safe_margin = max_flight_dist + max_cone_radius
        if self.margin < min_safe_margin:
            raise ValueError(f"Слишком маленький margin={self.margin:.1f} м нужно минимум {min_safe_margin:.1f} м")

        self.true_x = self.rng.uniform(self.margin, self.max_x - self.margin)
        self.true_y = self.rng.uniform(self.margin, self.max_y - self.margin)
        self.true_yaw = self.rng.uniform(-np.pi, np.pi)
        self.start_yaw = self.true_yaw

    def get_turn_rate(self):
        t = self.step_count * DT

        if self.route_name == "straight":
            return 0.0

        if self.route_name == "snake":
            return 0.25 * np.cos(2.0 * np.pi * t / 20.0)

        if self.route_name == "maneuver": #ai
            if 15.0 <= t < 17.0:
                return np.pi / 4.0      # поворот на +90 град за 2 сек
            if 30.0 <= t < 33.0:
                return -np.pi / 3.0     # разворот на -180 град за 3 сек
            return 0.0

        raise ValueError(f"Неизвестный маршрут: {self.route_name}") #ai

    def move(self, turn_rate=0.0):
        
        self.true_yaw += turn_rate * DT
        self.true_x += self.speed * np.cos(self.true_yaw) * DT
        self.true_y += self.speed * np.sin(self.true_yaw) * DT

        accel_std = self.noise_cfg["accel_std"]
        gyro_std = self.noise_cfg["gyro_std"]

        measured_dist = self.speed * DT + self.rng.normal(0.0, accel_std) #ai
        measured_dyaw = (turn_rate + self.rng.normal(0.0, gyro_std)) * DT #ai
        return float(measured_dist), float(measured_dyaw)

    def measure_vertical_lidar(self):
        ground = get_height(self.height_map, self.true_x, self.true_y)
        true_range = self.true_z - ground

        if true_range > LIDAR_MAX_RANGE or true_range < 0:
            return ERROR_LEN
        
        true_range = self.true_z - ground
        lidar_std = self.noise_cfg["lidar_std"]
        noise_range = true_range + self.rng.normal(0.0, lidar_std)
        return float(noise_range)

    def cast_ray(self, tilt_rad=0.0, azimuth_rad=0.0):
        angle = self.true_yaw + azimuth_rad
        R = self.true_z * np.tan(tilt_rad)

        end_x = self.true_x + R * np.cos(angle)
        end_y = self.true_y + R * np.sin(angle)

        xs = np.linspace(self.true_x, end_x, self.points)     # Делим отрезок дрона до пола на 600 точек
        ys = np.linspace(self.true_y, end_y, self.points)
        zs = np.linspace(self.true_z, 0.0, self.points)

        ground = get_height(self.height_map, xs, ys)  # Сверяем высоту луча с картой и берем первое пересечение
        hits = zs <= ground

        if not np.any(hits):
            return ERROR_LEN

        i = np.argmax(hits) # берет первый индекс появившегося True 
        true_range = np.sqrt(
            (xs[i] - self.true_x) ** 2
            + (ys[i] - self.true_y) ** 2
            + (zs[i] - self.true_z) ** 2
        )

        if true_range > LIDAR_MAX_RANGE:
            return ERROR_LEN
        
        lidar_std = self.noise_cfg["lidar_std"]
        noisy_range = true_range + self.rng.normal(0.0, lidar_std)
        return float(noisy_range)

    def measure_lidar(self):
        ranges = [self.measure_vertical_lidar()]
        for tilt_rad, azim_rad in zip(BEAM_TILTS[1:], BEAM_AZIMUTHS[1:]):
            ranges.append(self.cast_ray(tilt_rad, azim_rad))
        return np.array(ranges, dtype=np.float64)


    def get_initial_measurement(self): # функция для первого появления дрона (t = 0)
        return self.measure_lidar(), self.start_yaw
    

    def step(self):
        turn_rate = self.get_turn_rate()
        measured_dist, measured_dyaw = self.move(turn_rate=turn_rate)
        lidar_ranges = self.measure_lidar()
        self.step_count += 1
        return lidar_ranges, measured_dist, measured_dyaw

    def get_true_state(self):
        return self.true_x, self.true_y, self.true_z, self.true_yaw


if __name__ == "__main__":
    """
    H = generate_map()
    sim = DroneSimulator(H, route_name="snake", noise_cfg=DEFAULT_NOISE_CFG, seed=None)
    x0, y0, z0, yaw0 = sim.get_true_state()

    for _ in range(NUM_STEPS):
        lidar, dist, dyaw = sim.step()

    x1, y1, z1, yaw1 = sim.get_true_state()
    print(f"Старт: x={x0:.2f}, y={y0:.2f}, yaw={yaw0:.3f}")
    print(f"Финиш: x={x1:.2f}, y={y1:.2f}, yaw={yaw1:.3f}")
    print(f"Последний скан: центр={lidar[0]:.2f} м, конус min={lidar[1:].min():.2f} м, max={lidar[1:].max():.2f} м")
    """
    H = generate_map()
    sim = DroneSimulator(H, route_name="snake", noise_cfg=DEFAULT_NOISE_CFG)
    x0, y0, z0, yaw0 = sim.get_true_state()

    # Стартуем из точной начальной точки, чтобы увидеть чистый дрейф датчиков
    noisy_x, noisy_y, noisy_yaw = x0, y0, yaw0

    for _ in range(NUM_STEPS):
        lidar, dist, dyaw = sim.step()

        noisy_yaw += dyaw
        noisy_x += dist * np.cos(noisy_yaw)
        noisy_y += dist * np.sin(noisy_yaw)

    x1, y1, z1, yaw1 = sim.get_true_state()
    drift = np.hypot(x1 - noisy_x, y1 - noisy_y)

    print(f"Старт:        x={x0:.2f}, y={y0:.2f}, yaw={yaw0:.3f}")
    print(f"Истина финиш: x={x1:.2f}, y={y1:.2f}, yaw={yaw1:.3f}")
    print(f"По датчикам:  x={noisy_x:.2f}, y={noisy_y:.2f}, yaw={noisy_yaw:.3f}")
    print(f"Накопленная ошибка (дрейф): {drift:.2f} м")
