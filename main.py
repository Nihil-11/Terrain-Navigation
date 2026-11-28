import numpy as np

from config import NUM_STEPS, DEFAULT_NOISE_CFG
from simulator import generate_map, DroneSimulator, wrap_angle
from filter import ParticleFilter
from visualizer import DroneVisualizer

def run_simulation(
    route_name="snake",
    noise_cfg=DEFAULT_NOISE_CFG,
    seed=None,
    visualize=True,
    vis_every=10,
):
    H = generate_map() # создание карты высот

    sim = DroneSimulator(H, route_name=route_name, noise_cfg=noise_cfg, seed=seed) # запуск симуляции дрона
    pf = ParticleFilter(H) # запуск фильтра частиц
    pf_rng = np.random.default_rng(seed) # AI

    vis = DroneVisualizer(H) if visualize else None # запуск визуализации

    lidar_0, start_yaw = sim.get_initial_measurement() # шаг t = 0: начальный скан лидара 
    pf.initialize(start_yaw=start_yaw, rng=pf_rng) # спавн облака частиц
    pf.update_weights(lidar_0)

    est_state = pf.get_estimate() # оценку снимаем до resample пока веса еще не сброшены в 1/N
    pf.resample(rng=pf_rng)

    pos_errors = [] # отклонения по x, y 
    yaw_errors_deg = [] # отклонения от курса

    true_state = sim.get_true_state()
    tx, ty, _, tyaw = true_state # настоящее положение дрона
    ex, ey, _, eyaw = est_state # предположение о положении дрона
    pos_errors.append(float(np.hypot(tx - ex, ty - ey)))
    yaw_errors_deg.append(float(np.degrees(abs(wrap_angle(tyaw - eyaw)))))

    if vis is not None: vis.update(true_state, est_state=est_state, particles=pf.particles, step=0)

    # основной цикл полета
    for step in range(1, NUM_STEPS + 1):     
        lidar_ranges, measured_dist, measured_dyaw = sim.step()

        pf.predict(measured_dyaw=measured_dyaw, measured_dist=measured_dist, rng=pf_rng)
        pf.update_weights(lidar_ranges)

        est_state = pf.get_estimate() # оценку снимаем до resample пока веса еще не сброшены в 1/N
        pf.resample(rng=pf_rng) 

        # вывод погрешностей

        true_state = sim.get_true_state()
        tx, ty, _, tyaw = true_state
        ex, ey, _, eyaw = est_state

        err_pos = float(np.hypot(tx - ex, ty - ey))
        err_yaw = float(np.degrees(abs(wrap_angle(tyaw - eyaw))))
        pos_errors.append(err_pos)
        yaw_errors_deg.append(err_yaw)

        if step % 10 == 0 or step == NUM_STEPS:
            print(
                f"Шаг {step:4d}/{NUM_STEPS} | "
                f"Ошибка позиции: {err_pos:7.2f} м | "
                f"Ошибка курса: {err_yaw:5.2f} град"
            )

        if vis is not None and (step % vis_every == 0 or step == NUM_STEPS):
            vis.update(true_state, est_state=est_state, particles=pf.particles, step=step)

    if vis is not None: vis.finish()

if __name__ == "__main__":
    run_simulation(
        route_name="snake",
        noise_cfg=DEFAULT_NOISE_CFG,
        seed=None,
        visualize=True,
        vis_every=10,
    )