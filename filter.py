import numpy as np
from config import (
    BEAM_TILTS,
    BEAM_AZIMUTHS,
    NUM_PARTICLES,
    MAP_SIZE_METERS,
    FIXED_ALTITUDE_Z,
    DT,
    GYRO_NOISE_STD,
    ACCEL_NOISE_STD,
    MARGIN,
    ERROR_LEN,
)
from simulator import get_height, wrap_angle

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

        # состояние частицы [x, y, z, yaw]
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float64)
        self.particles[:, 2] = float(FIXED_ALTITUDE_Z)

        # каждой частице вес 1/N
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)

        self.yaw = 0.0 # начальный курс по умолчанию

    # равномерно распределяет координаты x, y для N частиц по площади карты (z фиксирована на 550м)
    # в пределах границ MARGIN, чтобы лучи не вышли за карту на первом шаге
    # задает равные веса 1 / N 
    def initialize(
        self,
        start_yaw: float = 0.0, # начальный курс
        init_yaw_std: float = 0.02, # шум
        rng: np.random.Generator | None = None,
    ):
        if rng is None: rng = np.random.default_rng()

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

    # предсказание, где дрон (сдвиг частиц с учетом шумов)
    def predict(
        self,
        measured_dyaw: float, # изменение курса за шаг dt
        measured_dist: float, # измеренная дистанция
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
        base_dist = float(measured_dist)
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

    # обновление весов через MSE и Гауссово правдоподобие на основе предсказания
    def update_weights(
        self,
        lidar_ranges: np.ndarray, # длины лучей с лидара
        sigma_eff: float = 120.0, # эффективное стандартное отклонение
        out_of_bounds_penalty: float = 10000.0, # сильный штраф за уход луча за карту (чтоб потом его откинуть)
    ):
        ranges = np.asarray(lidar_ranges, dtype=np.float64) # длины лучей с лидара: (51,)

        # отбрасываем ERROR_LEN (-1.0), неположительные и NaN/inf значения длин лучей
        valid_ranges = (
            np.isfinite(ranges)
            & (ranges != float(ERROR_LEN))
            & (ranges > 0.0)
        )

        # безопасная длина для тригонометрии (для невалидных лучей результат всё равно не пойдёт в среднее)
        safe_ranges = np.where(valid_ranges, ranges, 0.0)

        # горизонт. вынос и вертикальное смещение для 51 луча: (51,)
        horiz = safe_ranges * np.sin(BEAM_TILTS)
        dz = -safe_ranges * np.cos(BEAM_TILTS)

        # глобальные азимуты лучей для каждой частицы: (N, 51)
        angles = self.particles[:, 3:4] + BEAM_AZIMUTHS[None, :]

        # 3D координаты концов лучей для всех частиц: (N, 51)
        hit_x = self.particles[:, 0:1] + horiz[None, :] * np.cos(angles)
        hit_y = self.particles[:, 1:2] + horiz[None, :] * np.sin(angles)
        hit_z = self.particles[:, 2:3] + dz[None, :]

        # высота рельефа под каждым концом луча: (N, 51)
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

        # усредняем квадрат ошибки только по валидным лучам: (N,)
        valid_counts = np.sum(valid_ranges)
        if valid_counts == 0:
            mse = np.full(self.num_particles, float(out_of_bounds_penalty), dtype=np.float64)
        else:
            valid_sq_sum = np.sum(sq_diff[:, valid_ranges], axis=1)
            mse = valid_sq_sum / float(valid_counts)

        # AI --------

        # Гауссово правдоподобие в логарифмах с численной стабилизацией
        log_w = -mse / (2.0 * (float(sigma_eff) ** 2))
        log_w = log_w - np.max(log_w)
        raw_w = np.exp(log_w)

        # AI --------

        # распределение весов
        weight_sum = np.sum(raw_w)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights = raw_w / weight_sum

    # итоговая оценка состояния [x, y, z, yaw] по облаку частиц и их измененным весам
    def get_estimate(self) -> tuple[float, float, float, float]:
        weights = np.asarray(self.weights, dtype=np.float64)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        else: weights = weights / weight_sum

        est_x = float(np.sum(weights * self.particles[:, 0]))
        est_y = float(np.sum(weights * self.particles[:, 1]))
        est_z = float(np.sum(weights * self.particles[:, 2]))

        cos_mean = float(np.sum(weights * np.cos(self.particles[:, 3])))
        sin_mean = float(np.sum(weights * np.sin(self.particles[:, 3])))
        if np.hypot(cos_mean, sin_mean) < 1e-12:
            est_yaw = float(self.yaw)
        else: est_yaw = float(wrap_angle(np.arctan2(sin_mean, cos_mean)))

        return est_x, est_y, est_z, est_yaw

    # отбор гипотез и удаление слабых гипотез (пересобираем облако)
    # используется кум. сумма весов в отрезке [0; 1], на нее накладывается N точек (гипотез)
    # вложенный отрезок (текущая кум. сумма весов) не берется в рассмотрение, если в него не попала ни одна точка 
    def resample(
        self,
        rng: np.random.Generator | None = None,
    ):
        if rng is None: rng = np.random.default_rng()

        n = self.num_particles

        # защита на случай некорректной или вырожденной суммы весов
        weight_sum = np.sum(self.weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.full(n, 1.0 / n, dtype=np.float64)
        else: weights = self.weights / weight_sum

        cdf = np.cumsum(weights) # кум. сумма весов образует интервал [0; 1]
        cdf[-1] = 1.0 # правый конец интервала = 1 (пишу явно, чтобы не было например 0.99998 тк float)

        # случайный сдвиг в первом интервале [0, 1/n)
        u0 = rng.uniform(0.0, 1.0 / n)

        # позиция на интервале для N частиц с учетом сдвига u0 и шагом 1/n 
        positions = u0 + np.arange(n, dtype=np.float64) / n 

        # для каждой точки из positions берется индекс интервала из cdf, в который эта точка попала  
        # side="right" для того, чтобы точка на конце интервала относилась к след. интервалу
        indices = np.searchsorted(cdf, positions, side="right")
        indices = np.clip(indices, 0, n - 1)

        resampled = self.particles[indices].copy()
        resampled[:, 0] = np.clip(resampled[:, 0], 0.0, MAP_SIZE_METERS) # ограничение координат для страховки
        resampled[:, 1] = np.clip(resampled[:, 1], 0.0, MAP_SIZE_METERS) # ограничение координат для страховки
        resampled[:, 2] = float(FIXED_ALTITUDE_Z) # высота остается 550м
        resampled[:, 3] = wrap_angle(resampled[:, 3]) # приводим курс частиц к каноническому диапазону углов

        self.particles = resampled # сохраняем новое состояния частиц
        self.weights.fill(1.0 / n) # снова закидываем равномерные веса для след. итерации изменения весов