import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from config import (
    MAP_SIZE_METERS,
    FIXED_ALTITUDE_Z,
    LIDAR_TILT_RAD,
    NUM_STEPS,
    DEFAULT_NOISE_CFG,
)


class DroneVisualizer:
    def __init__(self, height_map):
        plt.ion()  # Включаем интерактивный режим, чтобы окно не блокировало цикл программы
        self.fig, self.ax_drone = plt.subplots(figsize=(8, 7))

        # Рисуем карту один раз: берем каждый 4-й пиксель для скорости и транспонируем под оси (X, Y)
        preview_map = height_map[::4, ::4].T
        img = self.ax_drone.imshow(
            preview_map,
            origin="lower",
            extent=[0.0, MAP_SIZE_METERS, 0.0, MAP_SIZE_METERS],
            cmap="terrain",
        )
        self.fig.colorbar(img, ax=self.ax_drone, label="Высота рельефа (м)")

        self.ax_drone.set_title("Реальное движение дрона")
        self.ax_drone.set_xlabel("X (м)")
        self.ax_drone.set_ylabel("Y (м)")
        self.ax_drone.set_xlim(0.0, MAP_SIZE_METERS)
        self.ax_drone.set_ylim(0.0, MAP_SIZE_METERS)

        # История координат для отрисовки хвоста траектории
        self.path_x = []
        self.path_y = []

        # Создаем пустые графические объекты, которые потом будем просто сдвигать
        (self.trail_line,) = self.ax_drone.plot(
            [], [], color="red", linewidth=1.5, label="Траектория"
        )
        (self.drone_dot,) = self.ax_drone.plot(
            [],
            [],
            marker="o",
            color="yellow",
            markeredgecolor="black",
            markersize=7,
            label="Дрон",
        )
        (self.heading_line,) = self.ax_drone.plot([], [], color="black", linewidth=2.0)

        # Пунктирный круг: примерный радиус конуса лидара на земле (~200 м)
        cone_radius = FIXED_ALTITUDE_Z * np.tan(LIDAR_TILT_RAD)
        self.cone_circle = Circle(
            (0.0, 0.0),
            cone_radius,
            fill=False,
            edgecolor="cyan",
            linestyle="--",
            linewidth=1.2,
            label="Зона лидара",
        )
        self.ax_drone.add_patch(self.cone_circle)

        self.ax_drone.legend(loc="upper right")
        self.fig.tight_layout()

    def update(self, x, y, yaw, step=None):
        self.path_x.append(x)
        self.path_y.append(y)

        # Обновляем хвост и точку самого дрона
        self.trail_line.set_data(self.path_x, self.path_y)
        self.drone_dot.set_data([x], [y])

        # Короткая стрелка-линия длиной 120 м, показывающая направление носа дрона
        nose_len = 120.0
        self.heading_line.set_data(
            [x, x + nose_len * np.cos(yaw)],
            [y, y + nose_len * np.sin(yaw)],
        )

        # Сдвигаем центр круга лидара под текущие координаты дрона
        self.cone_circle.center = (x, y)

        if step is not None:
            self.ax_drone.set_title(f"Реальное движение дрона (шаг {step})")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def finish(self):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    from simulator import generate_map, DroneSimulator

    H = generate_map()
    sim = DroneSimulator(H, route_name="snake", noise_cfg=DEFAULT_NOISE_CFG)
    vis = DroneVisualizer(H)

    x0, y0, z0, yaw0 = sim.get_true_state()
    vis.update(x0, y0, yaw0, step=0)

    for step in range(1, NUM_STEPS + 1):
        sim.step()
        x, y, z, yaw = sim.get_true_state()

        # Обновляем картинку каждые 5 шагов, чтобы анимация шла бодро и без лагов
        if step % 10== 0:
            vis.update(x, y, yaw, step=step)

    vis.finish()