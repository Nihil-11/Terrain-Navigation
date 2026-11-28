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
ОБЩИЙ АЛГОРИТМ РАБОТЫ:
1. __init__(height_map): один раз открывает окно с двумя панелями (слева дрон,
   справа фильтр), рисует карту высот как фон и создает пустые заготовки объектов.
2. update(true_state, est_state, particles, step): не перерисовывает фон,
   а только меняет координаты у готовых объектов на обеих панелях и обновляет кадр.
3. finish(): выключает интерактивный режим и держит окно открытым после цикла.
4. if __name__ == "__main__": тестовый запуск одного симулятора без фильтра
   (обновление кадра идет раз в 10 шагов через step % 10 == 0).

ОСНОВНЫЕ ПЕРЕМЕННЫЕ И ОБЪЕКТЫ (внутри DroneVisualizer):
- fig (Figure): само окно целиком (размер, рамка, холст canvas).
- ax_drone (Axes): левое поле координат для реального движения дрона.
- ax_pf (Axes): правое поле координат для работы фильтра частиц.
- preview_map: облегченная копия карты для фона (каждый 4-й пиксель).
- img_pf: графический объект фоновой карты (к нему привязана общая шкала высот).
- path_x, path_y: списки прошлых координат дрона для красного хвоста слева.
- est_path_x, est_path_y: списки прошлых оценок фильтра для пути оценки справа.
- trail_line: красная линия истинного пути (Line2D).
- drone_dot: желтая точка текущего положения дрона слева.
- heading_line: черный отрезок длиной nose_len (100 м), куда смотрит нос дрона.
- cone_radius: радиус конуса лидара на уровне земли (~200.2 м).
- cone_circle: геометрическая окружность (Circle) зоны обзора лидара.
- particles_dots: облако всех частиц фильтра справа (точки без соединяющей линии).
- est_trail_line: фиолетовый пунктир пройденного пути по оценке фильтра.
- est_dot: красный крестик текущей оценки позиции фильтра.
- true_dot_pf: желтая точка реального дрона справа (для сравнения с оценкой).

КОМАНДЫ MATPLOTLIB (ОКНО И АНИМАЦИЯ):
- plt.ion(): включает интерактивный режим, окно не тормозит выполнение цикла for.
- plt.ioff(): выключает интерактивный режим обратно в конце программы.
- plt.show(): останавливает скрипт и держит финальное окно открытым до закрытия.
- plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True): создает окно fig
  с двумя полями (1 ряд, 2 столбца) и авто-подгонкой отступов вместо tight_layout.
- fig.canvas.draw_idle(): сообщает движку окна, что данные изменились и нужен новый кадр.
- plt.pause(0.001): пауза 1 мс, чтобы ОС успела отрисовать кадр и окно не зависло.

КОМАНДЫ MATPLOTLIB (КАРТА, ОСИ, ЛИНИИ И ФИГУРЫ):
- ax.imshow(data, origin="lower", extent=[...], cmap="terrain", alpha=...):
  рисует 2D-матрицу как цветную картинку.
  * origin="lower": ставит ноль (0, 0) в левый нижний угол (а не верхний).
  * extent=[0, 4075, 0, 4075]: привязывает границы картинки к метрам.
  * cmap="terrain": цветовая палитра рельефа (синий -> зеленый -> коричневый -> белый).
  * alpha=0.8: прозрачность (0 прозрачно, 1 сплошной цвет), приглушает фон под частицами.
- fig.colorbar(img, ax=[ax1, ax2], shrink=0.85, label=...): общая шкала высот
  справа от обеих панелей (shrink чуть укорачивает ее по высоте).
- ax.set_title / set_xlabel / set_ylabel: заголовок графика и подписи осей.
- ax.set_xlim / set_ylim: жестко фиксируют границы обзора, чтобы камера не прыгала.
- ax.plot([], [], ...): создает пустой объект линии заранее, чтобы потом двигать его.
  * color: цвет линии или заливки маркера.
  * linewidth: толщина линии.
  * linestyle: стиль линии ("--" пунктир, "none" не соединять точки линией).
  * marker: форма точки ("o" кружок, "." мелкая точка, "X" крестик).
  * markersize: размер маркера на экране, markeredgecolor: цвет его обводки.
  * zorder: порядок слоев (чем больше число, тем выше объект рисуется поверх других).
  * label: текст для таблички легенды.
- Circle((0.0, 0.0), r, fill=False, edgecolor=..., linestyle="--"):
  создает фигуру окружности с точным радиусом r в метрах карты.
  * (0.0, 0.0): временный центр-заглушка до первого вызова update.
  * fill=False: прозрачная внутри (только контур).
- ax.add_patch(circle): прикрепляет созданную фигуру Circle на координатное поле.
- ax.legend(loc="upper right", fontsize="small"): табличка легенды в правом верхнем углу.
- line.set_data(xs, ys): полностью заменяет старые координаты линии или точек на новые.
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

        # ИЗМЕНЕНО: два поля рядом, размер под экран и авто-выравнивание
        self.fig, (self.ax_drone, self.ax_pf) = plt.subplots(
            1, 2, figsize=(12, 5.5), constrained_layout=True
        )

        preview_map = height_map[::4, ::4].T
        extent = [0.0, MAP_SIZE_METERS, 0.0, MAP_SIZE_METERS]

        # ЛЕВАЯ ПАНЕЛЬ: РЕАЛЬНЫЙ ДРОН
        self.ax_drone.imshow(
            preview_map,
            origin="lower",
            extent=extent,
            cmap="terrain",
        )

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
            [], [], color="red", linewidth=1.5, label="Истинный путь"
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
        self.ax_drone.legend(loc="upper right", fontsize="small")

        # ДОБАВЛЕНО: ПРАВАЯ ПАНЕЛЬ (ФИЛЬТР ЧАСТИЦ)
        img_pf = self.ax_pf.imshow(
            preview_map,
            origin="lower",
            extent=extent,
            cmap="terrain",
        )
        # Общая шкала высот для обеих карт, чтобы они были строго одного размера
        self.fig.colorbar(
            img_pf,
            ax=[self.ax_drone, self.ax_pf],
            label="Высота рельефа (м)",
        )

        self.ax_pf.set_title("Фильтр частиц")
        self.ax_pf.set_xlabel("X (м)")
        self.ax_pf.set_ylabel("Y (м)")
        self.ax_pf.set_xlim(0.0, MAP_SIZE_METERS)
        self.ax_pf.set_ylim(0.0, MAP_SIZE_METERS)

        self.est_path_x = []
        self.est_path_y = []

        (self.particles_dots,) = self.ax_pf.plot(
            [],
            [],
            linestyle="none",
            marker=".",
            color="blue",
            markersize=2.5,
            zorder=2,
            label="Частицы",
        )
        
        # zorder=5 держит желтую точку поверх красного крестика, чтобы оба маркера были видны
        (self.true_dot_pf,) = self.ax_pf.plot(
            [],
            [],
            marker="o",
            color="yellow",
            markeredgecolor="black",
            markersize=6,
            zorder=5,
            label="Истина",
        )
        self.ax_pf.legend(loc="upper right", fontsize="small")
        # КОНЕЦ ДОБАВЛЕННОГО БЛОКА 

    def update(self, true_state, est_state=None, particles=None, step=None):     # ИЗМЕНЕНО: принимаем кортежи состояний и массив частиц целиком
        x, y, _, yaw = true_state

        self.path_x.append(x)
        self.path_y.append(y)

        # Обновляем хвост и точку самого дрона слева
        self.trail_line.set_data(self.path_x, self.path_y)
        self.drone_dot.set_data([x], [y])

        nose_len = 100.0
        self.heading_line.set_data(
            [x, x + nose_len * np.cos(yaw)],
            [y, y + nose_len * np.sin(yaw)],
        )

        self.cone_circle.center = (x, y)

        if step is not None:
            self.ax_drone.set_title(f"Реальное движение дрона (шаг {step})")

        # ДОБАВЛЕНО: обновление правой панели (фильтр частиц)
        self.true_dot_pf.set_data([x], [y])

        if particles is not None:
            self.particles_dots.set_data(particles[:, 0], particles[:, 1])


        # КОНЕЦ ДОБАВЛЕННОГО БЛОКА

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def finish(self):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    from simulator import generate_map, DroneSimulator

    H = generate_map()
    sim = DroneSimulator(H, route_name="snake", noise_cfg=DEFAULT_NOISE_CFG)
    vis = DroneVisualizer(H)

    vis.update(sim.get_true_state(), step=0)

    for step in range(1, NUM_STEPS + 1):
        sim.step()

        # Обновляем каждые 10 шагов
        if step % 10 == 0:
            vis.update(sim.get_true_state(), step=step)

    vis.finish()