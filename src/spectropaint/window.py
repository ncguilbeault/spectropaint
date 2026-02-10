import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class Window(tk.Tk):

    def __init__(self):
        # Create the main window
        super().__init__()
        self.title("Spectrogram Painter")
        self.geometry("300x200")

        # Define image size
        self.image_width = 25
        self.image_height = 25

        # Create matplotlib figure to display data
        fig = Figure(figsize=(5, 3), dpi=200)
        self.ax = fig.add_subplot()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.vlines(
            [i / self.image_width for i in range(self.image_width)],
            0,
            1,
            color="black",
            alpha=0.2,
            lw=1,
        )
        self.ax.hlines(
            [i / self.image_height for i in range(self.image_height)],
            0,
            1,
            color="black",
            alpha=0.2,
            lw=1,
        )

        self.image_data = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint8
        )

        self.image_coords = np.meshgrid(
            np.linspace(0, 1, self.image_width),
            np.linspace(0, 1, self.image_height),
        )
        self.image_coords = np.stack(self.image_coords, axis=-1)

        self.imshow = self.ax.imshow(
            self.image_data,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="gray_r",
            vmin=0,
            vmax=255,
        )

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.pause_refresh)

        self.scale = 10
        self.radius = 1
        self.pos = np.array([0.0, 0.0])
        self.refresh_rate = int(1 / 10 * 1000)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.running = False
        self.job_id = None

        self.scale_tk = tk.DoubleVar(value=self.scale)

        scale_slider = tk.Scale(
            self,
            label="Intensity",
            from_=1,
            to=40,
            resolution=0.5,
            orient="horizontal",
            variable=self.scale_tk,
        )
        scale_slider.pack(fill="x", padx=10, pady=10)

        self.radius_tk = tk.DoubleVar(value=self.radius)

        radius_slider = tk.Scale(
            self,
            label="Radius",
            from_=0.25,
            to=5,
            resolution=0.25,
            orient="horizontal",
            variable=self.radius_tk,
        )
        radius_slider.pack(fill="x", padx=10, pady=10)

    def schedule_next(self):
        self.job_id = self.after(self.refresh_rate, self.refresh_canvas)

    def pause_refresh(self, value):
        button = value.button
        if button == 1 and self.running:
            self.running = False
            if self.job_id is not None:
                self.after_cancel(self.job_id)
                self.job_id = None

    def on_close(self):
        self.running = False
        if self.job_id is not None:
            self.after_cancel(self.job_id)
            self.job_id = None
        self.destroy()

    def refresh_canvas(self):
        if not self.running:
            self.job_id = None
            return

        distances = np.linalg.norm(self.image_coords - self.pos, axis=-1)
        mask = distances <= self.radius_tk.get() / max(
            self.image_width, self.image_height
        )
        self.image_data[mask] = np.minimum(
            self.image_data[mask] + self.scale_tk.get(), 255
        )
        self.imshow.set_data(self.image_data)
        self.canvas.draw_idle()
        self.schedule_next()

    def on_mouse_move(self, value):
        # print(f"Data: {value}")
        if value.xdata is None or value.ydata is None:
            return
        self.pos = np.array([value.xdata, value.ydata])
        button = value.button
        if button == 1 and not self.running:
            self.running = True
            self.schedule_next()
