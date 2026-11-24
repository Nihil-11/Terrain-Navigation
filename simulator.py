import numpy as np
from config import TERRAIN_FILE, MAP_SIZE, MAP_RESOLUTION, DT, LIDAR_MAX_RANGE

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
    def __init__(self, height_map, margin=1000.0, start_z=550.0, speed = 15, seed=None):
        self.height_map = height_map
        self.rng = np.random.default_rng(seed)
        self.speed = speed

        max_x = height_map.shape[0] * MAP_RESOLUTION
        max_y = height_map.shape[1] * MAP_RESOLUTION

        self.true_x = self.rng.uniform(margin, max_x - margin)
        self.true_y = self.rng.uniform(margin, max_y - margin)
        self.true_z = start_z
        self.true_yaw = self.rng.uniform(-np.pi, np.pi)

    def move(self, turn_rate=0.0):
        self.true_yaw += turn_rate * DT
        self.true_x += self.speed * np.cos(self.true_yaw) * DT
        self.true_y += self.speed * np.sin(self.true_yaw) * DT

        measured_dist = self.speed * DT + self.rng.normal(0.0, 0.15) #ai
        measured_dyaw = turn_rate * DT + self.rng.normal(0.0, 0.01) #ai
        return measured_dist, measured_dyaw

    def measure_vertical_lidar(self):
        ground = get_height(self.height_map, self.true_x, self.true_y)
        true_range = self.true_z - ground
        noise_range = true_range + self.rng.normal(0.0, 0.5)
        return float(np.clip(noise_range, 0.0, LIDAR_MAX_RANGE))

    def get_true_state(self):
        return self.true_x, self.true_y, self.true_z, self.true_yaw


if __name__ == "__main__":
    H = generate_map()
    sim = DroneSimulator(H)

    for step in range(5):
        dist, dyaw = sim.move(turn_rate=0.1)
        lidar = sim.measure_vertical_lidar()
        x, y, z, yaw = sim.get_true_state()
        true_range = z - get_height(H, x, y)

        print(
            f"Шаг {step}: путь={dist:.2f} м, поворот={dyaw:.4f} рад, "
            f"лидар={lidar:.2f} м (реально {true_range:.2f} м)"
        )