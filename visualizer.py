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

"""
1. __init__(height_map): один раз открывает окно, рисует тяжелую карту высот
   как фон и создает пустые заготовки объектов (линии, точка, круг).
2. update(x, y, yaw, step): не перерисовывает фон, а только меняет координаты
   у готовых объектов и обновляет кадр.
3. finish(): выключает интерактивный режим и держит окно открытым после цикла.
4. if __name__ == "__main__": тестовый запуск связки симулятора и окна
   (обновление кадра идет раз в 10 шагов через step % 10 == 0).

ОСНОВНЫЕ ПЕРЕМЕННЫЕ И ОБЪЕКТЫ (внутри DroneVisualizer):
- fig (Figure): само окно целиком (размер, рамка, холст canvas).
- ax_drone (Axes): поле координат внутри окна с осями X и Y, где все рисуется.
- preview_map: облегченная копия карты для фона (каждый 4-й пиксель).
- img: графический объект фоновой карты (нужен, чтобы привязать к нему colorbar).
- path_x, path_y: списки всех прошлых координат дрона для рисования красного хвоста.
- trail_line: объект красной линии пройденного пути (Line2D).
- drone_dot: объект желтой точки текущего положения дрона (Line2D с маркером).
- heading_line: объект черного отрезка длиной nose_len (100 м), куда смотрит нос.
- cone_radius: радиус конуса лидара на уровне земли (~200.2 м).
- cone_circle: геометрическая окружность (Circle) зоны обзора лидара.

КОМАНДЫ MATPLOTLIB (ОКНО И АНИМАЦИЯ):
- plt.ion(): включает интерактивный режим, окно не тормозит выполнение цикла for.
- plt.ioff(): выключает интерактивный режим обратно в конце программы.
- plt.show(): останавливает скрипт и держит финальное окно открытым до закрытия.
- plt.subplots(figsize=(8, 7)): создает окно fig и координатное поле ax
  размером 8x7 дюймов.
- fig.tight_layout(): подгоняет отступы вокруг графика, чтобы подписи не обрезались.
- fig.canvas.draw_idle(): сообщает движку окна, что данные изменились и нужен новый кадр.
- plt.pause(0.001): пауза 1 мс, чтобы ОС успела отрисовать кадр и окно не зависло.

КОМАНДЫ MATPLOTLIB (КАРТА, ОСИ, ЛИНИИ И ФИГУРЫ):
- ax.imshow(data, origin="lower", extent=[...], cmap="terrain"):
  рисует 2D-матрицу как цветную картинку.
  * origin="lower": ставит ноль (0, 0) в левый нижний угол (а не верхний).
  * extent=[0, 4075, 0, 4075]: привязывает границы картинки к метрам.
  * cmap="terrain": цветовая палитра рельефа (синий -> зеленый -> коричневый -> белый).
- fig.colorbar(img, ax=..., label=...): добавляет сбоку шкалу высот по цветам img.
- ax.set_title / set_xlabel / set_ylabel: заголовок графика и подписи осей.
- ax.set_xlim / set_ylim: жестко фиксируют границы обзора, чтобы камера не прыгала.
- ax.plot([], [], ...): создает пустой объект линии заранее, чтобы потом двигать его.
  * color: цвет линии или заливки маркера.
  * linewidth: толщина линии.
  * marker="o": форма точки (кружок), markersize: размер точки на экране.
  * markeredgecolor: цвет обводки точки.
  * label: текст для таблички легенды.
- Circle((0.0, 0.0), r, fill=False, edgecolor=..., linestyle="--"):
  создает фигуру окружности с точным радиусом r в метрах карты.
  * (0.0, 0.0): временный центр-заглушка до первого вызова update.
  * fill=False: прозрачная внутри (только контур), linestyle="--": пунктир.
- ax.add_patch(circle): прикрепляет созданную фигуру Circle на координатное поле.
- ax.legend(loc="upper right"): рисует табличку легенды в правом верхнем углу
  из всех объектов, у которых задан параметр label.
- line.set_data(xs, ys): полностью заменяет старые координаты линии на новые.
- cone_circle.center = (x, y): переносит центр круга в новые координаты.

PYTHON/NUMPY:
- height_map[::4, ::4].T: срез ::4 берет каждый 4-й пиксель для скорости отрисовки,
  а .T (транспонирование) меняет местами X и Y, чтобы карта не легла боком.
- (self.trail_line,) = ax.plot(...): plot всегда возвращает список линий [line],
  запятая в скобках достает саму линию из списка (аналог ...plot(...)[0]).
- set_data([x], [y]): квадратные скобки вокруг x и y обязательны, потому что
  set_data всегда ждет списки координат, даже если точка всего одна.
"""

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

        # Пунктирный круг: примерный радиус конуса лидара на земле 200 м
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
        nose_len = 100.0
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
    sim = DroneSimulator(H, route_name="maneuver", noise_cfg=DEFAULT_NOISE_CFG)
    vis = DroneVisualizer(H)

    x0, y0, z0, yaw0 = sim.get_true_state()
    vis.update(x0, y0, yaw0, step=0)

    for step in range(1, NUM_STEPS + 1):
        sim.step()
        x, y, z, yaw = sim.get_true_state()

        # Обновляем картинку каждые 5 шагов
        if step % 10== 0:
            vis.update(x, y, yaw, step=step)

    vis.finish()