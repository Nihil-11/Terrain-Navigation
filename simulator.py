import numpy as np
from config import (
    TERRAIN_FILE,
    MAP_SIZE,
    MAP_RESOLUTION,
    DT,
    LIDAR_MAX_RANGE,
    BEAM_TILTS,
    BEAM_AZIMUTHS,  
    DEFAULT_NOISE_CFG,
    ZERO_NOISE_CFG,
)

def generate_map(size=MAP_SIZE, seed=10):
    H = np.load(TERRAIN_FILE)
    H = H.astype(np.float64) # распаковываем из float34 в float64
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
    def __init__(self, height_map, noise_cfg=None, margin=1000.0, start_z=550.0, speed = 15, seed=None):
        self.height_map = height_map
        self.noise_cfg = DEFAULT_NOISE_CFG if noise_cfg is None else noise_cfg #ai
        self.rng = np.random.default_rng(seed)
        self.speed = speed

        max_x = height_map.shape[0] * MAP_RESOLUTION
        max_y = height_map.shape[1] * MAP_RESOLUTION

        self.true_x = self.rng.uniform(margin, max_x - margin)
        self.true_y = self.rng.uniform(margin, max_y - margin)
        self.true_z = float(start_z)
        self.true_yaw = self.rng.uniform(-np.pi, np.pi)
        self.start_yaw = self.true_yaw

        self.cone_tilt = 20.0
        self.cone_azimuths = np.linspace(0.0, 360.0, 50, endpoint=False)

    def move(self, turn_rate=0.0):
        self.true_yaw += turn_rate * DT
        self.true_x += self.speed * np.cos(self.true_yaw) * DT
        self.true_y += self.speed * np.sin(self.true_yaw) * DT

        accel_std = self.noise_cfg["accel_std"]
        gyro_std = self.noise_cfg["gyro_std"]

        measured_dist = self.speed * DT + self.rng.normal(0.0, accel_std) #ai
        measured_dyaw = turn_rate * DT + self.rng.normal(0.0, gyro_std) #ai
        return measured_dist, measured_dyaw

    def measure_vertical_lidar(self):
        ground = get_height(self.height_map, self.true_x, self.true_y)
        true_range = self.true_z - ground
        lidar_std = self.noise_cfg["lidar_std"]
        noise_range = true_range + self.rng.normal(0.0, lidar_std)
        return float(np.clip(noise_range, 0.0, LIDAR_MAX_RANGE))

    def cast_ray(self, tilt_rad=0.0, azimuth_rad=0.0):
        angle = self.true_yaw + azimuth_rad
        R = self.true_z * np.tan(tilt_rad)

        end_x = self.true_x + R * np.cos(angle)
        end_y = self.true_y + R * np.sin(angle)

        xs = np.linspace(self.true_x, end_x, 600)     # Делим отрезок дрона до пола на 600 точек
        ys = np.linspace(self.true_y, end_y, 600)
        zs = np.linspace(self.true_z, 0.0, 600)

        ground = get_height(self.height_map, xs, ys)  # Сверяем высоту луча с картой и берем первое пересечение
        hits = zs <= ground

        if np.any(hits):
            i = np.argmax(hits)
            true_range = np.sqrt(
                (xs[i] - self.true_x) ** 2
                + (ys[i] - self.true_y) ** 2
                + (zs[i] - self.true_z) ** 2
            )
        else:
            true_range = LIDAR_MAX_RANGE

        noisy_range = true_range + self.rng.normal(0.0, 0.5)
        return float(np.clip(noisy_range, 0.0, LIDAR_MAX_RANGE))

    def measure_lidar(self):
        ranges = [self.measure_vertical_lidar()]
        for tilt_rad, azim_rad in zip(BEAM_TILTS[1:], BEAM_AZIMUTHS[1:]):
            ranges.append(self.cast_ray(tilt_rad, azim_rad))
        return np.array(ranges, dtype=np.float64)

    def get_initial_measurement(self):
        return self.measure_lidar(), self.start_yaw

    def get_true_state(self):
        return self.true_x, self.true_y, self.true_z, self.true_yaw


if __name__ == "__main__":
    H = generate_map()
    print(f"Карта: shape={H.shape}, min={H.min():.2f} м, max={H.max():.2f} м")

    sim = DroneSimulator(H, noise_cfg=ZERO_NOISE_CFG, seed=42)
    scan, start_yaw = sim.get_initial_measurement()
    x, y, z, yaw = sim.get_true_state()
    true_center = z - get_height(H, x, y)

    print(f"Центральный луч: {scan[0]:.2f} м (истина: {true_center:.2f} м)")
    print(f"Конус (50 лучей): min={scan[1:].min():.2f} м, max={scan[1:].max():.2f} м")