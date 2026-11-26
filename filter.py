import numpy as np
from config import (
    BEAM_TILTS,
    BEAM_AZIMUTHS,
    NUM_PARTICLES,
    MAP_SIZE_METERS,
    FIXED_ALTITUDE_Z,
    DRONE_FIXED_SPEED,
    DT,
    GYRO_NOISE_STD,
    ACCEL_NOISE_STD,
    MARGIN,
)
from simulator import get_height, wrap_angle

# считаем синусы и косинусы наклонов для скорости
_SIN_TILTS = np.sin(BEAM_TILTS)
_COS_TILTS = np.cos(BEAM_TILTS)

# фильтр частиц для оценки состояния дрона по карте высот
class ParticleFilter:
    def __init__(
        self,
        height_map: np.ndarray, # карта высот
        num_particles: int = NUM_PARTICLES, # число частиц
    ):
        if num_particles <= 0:
            raise ValueError("num_particles должно быть положительным числом")

        self.height_map = np.asarray(height_map, dtype=np.float64) # матрица высот
        self.num_particles = int(num_particles) # число частиц
        self.fixed_speed = float(DRONE_FIXED_SPEED) # постоянная скорость 15 м/с

        # состояние частицы [x, y, z, yaw]
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float64)
        self.particles[:, 2] = float(FIXED_ALTITUDE_Z)

        # каждой частице вес 1/N
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)

        self.yaw = 0.0 # начальный курс по умолчанию

    # равномерно распределяет координаты x, y для N частиц по площади карты (z фиксирована на 550м)
    # задает равные веса 1 / N 
    def initialize(
        self,
        start_yaw: float = 0.0, # начальный курс
        init_yaw_std: float = 0.02, # шум
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        if MARGIN is not None:
            m = float(MARGIN)
            if m < 0.0 or 2.0 * m >= MAP_SIZE_METERS:
                raise ValueError("Некорректное значение margin для размера карты")
            low, high = m, MAP_SIZE_METERS - m
        else:
            low, high = 0.0, MAP_SIZE_METERS

        self.yaw = float(wrap_angle(float(start_yaw))) # чтобы угол yaw был в полуинтервале [-pi, pi)
        yaw_noise = rng.normal(0.0, init_yaw_std, size=self.num_particles) if init_yaw_std > 0.0 else 0.0

        self.particles[:, 0] = rng.uniform(low, high, size=self.num_particles) # xi
        self.particles[:, 1] = rng.uniform(low, high, size=self.num_particles) # yi
        self.particles[:, 2] = float(FIXED_ALTITUDE_Z) # zi
        self.particles[:, 3] = wrap_angle(self.yaw + yaw_noise) # yawi (yaw частицы)

        self.weights.fill(1.0 / self.num_particles) # обновление весов для многократ. вызова initialize

    # обновление весов перед предсказанием
    def update_weights(
        self,
        lidar_ranges: np.ndarray, # длины лучей с лидара
        sigma_eff: float = 3.0, # эффективное стандартное отклонение
        out_of_bounds_penalty: float = 100.0, # сильный штраф за уход луча за карту (чтоб потом его откинуть) 
    ):
        ranges = np.asarray(lidar_ranges, dtype=np.float64) # копия длин лучей с лидара

        # горизонт. вынос и вертикальное смещение для 51 луча: (51,)
        horiz = ranges * _SIN_TILTS
        dz = -ranges * _COS_TILTS

        # глобальные азимуты лучей для каждой частицы: (N, 51)
        angles = self.particles[:, 3:4] + BEAM_AZIMUTHS[None, :]

        # 3D координаты концов лучей для всех частиц: (N, 51)
        hit_x = self.particles[:, 0:1] + horiz[None, :] * np.cos(angles)
        hit_y = self.particles[:, 1:2] + horiz[None, :] * np.sin(angles)
        hit_z = self.particles[:, 2:3] + dz[None, :]

        # высота рельефа под каждым концом луча (N, 51)
        map_h = get_height(self.height_map, hit_x, hit_y)

        # квадрат разницы высот + штраф за выход конца луча за [0, MAP_SIZE_METERS]
        sq_diff = (hit_z - map_h) ** 2
        oob_mask = (
            (hit_x < 0.0)
            | (hit_x > MAP_SIZE_METERS)
            | (hit_y < 0.0)
            | (hit_y > MAP_SIZE_METERS)
        )
        sq_diff = np.where(oob_mask, sq_diff + out_of_bounds_penalty, sq_diff)

        # MSE по 51 лучу для каждой частицы: (N,)
        mse = np.mean(sq_diff, axis=1)

        # Гауссово правдоподобие в логарифмах с численной стабилизацией
        log_w = -mse / (2.0 * (float(sigma_eff) ** 2))
        log_w = log_w - np.max(log_w)
        raw_w = np.exp(log_w)

        # распределение весов
        weight_sum = np.sum(raw_w)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            self.weights.fill(1.0 / self.num_particles)
        else: self.weights = raw_w / weight_sum

    # предсказание, где дрон
    def predict(
        self,
        measured_dyaw: float, # изменение курса за шаг dt
        dt: float = DT, # шаг времени
        gyro_noise_std: float = GYRO_NOISE_STD, # стандартное отклонение шума гироскопа (угловой скорости)
        accel_noise_std: float = ACCEL_NOISE_STD, # стандартное отклонение шума величины шага за такт (шум уск.)
        rng: np.random.Generator | None = None,
    ):
        if rng is None: rng = np.random.default_rng()

        # курс для каждой частицы с шумом гироскопа
        yaw_std = gyro_noise_std * dt
        yaw_noise = (
            rng.normal(0.0, yaw_std, size=self.num_particles)
            if yaw_std > 0.0
            else 0.0
        )
        new_yaw = wrap_angle(self.particles[:, 3] + measured_dyaw + yaw_noise)

        # шаг за dt при фиксированной скорости + индивидуальный шум для каждой частицы
        base_dist = self.fixed_speed * dt
        dist_noise = (
            rng.normal(0.0, accel_noise_std, size=self.num_particles)
            if accel_noise_std > 0.0
            else 0.0
        )
        step_dist = base_dist + dist_noise

        # сдвиг в плоскости XY с ограничением по карте, z фиксирована на 550 м
        self.particles[:, 0] = np.clip(
            self.particles[:, 0] + step_dist * np.cos(new_yaw),
            0.0,
            MAP_SIZE_METERS,
        )
        self.particles[:, 1] = np.clip(
            self.particles[:, 1] + step_dist * np.sin(new_yaw),
            0.0,
            MAP_SIZE_METERS,
        )
        self.particles[:, 2] = float(FIXED_ALTITUDE_Z)
        self.particles[:, 3] = new_yaw

        self.yaw = float(wrap_angle(self.yaw + measured_dyaw)) # изменение курса