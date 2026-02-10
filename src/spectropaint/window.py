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
        self.image_width = 100
        self.image_height = 100

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
            alpha=0.2,
            zorder=2,
        )

        self.imshow_underlays = []
        self.spectrogram_playback_pos = 0
        self.painted_playback_pos = 0

        self.ax.vlines(self.spectrogram_playback_pos, 0, 1, color="red", lw=2, zorder=3)
        self.ax.vlines(self.painted_playback_pos, 0, 1, color="green", lw=2, zorder=3)

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
        self.paint_job = None

        self.hide_spectrogram = tk.BooleanVar(value=False)
        hide_checkbox = tk.Checkbutton(
            self,
            text="Hide Spectrogram",
            variable=self.hide_spectrogram,
            command=self.update_spectrogram_visibility,
        )
        hide_checkbox.pack(fill="x", side="left", padx=10, pady=10)

        self.play_spectrogram = tk.BooleanVar(value=False)
        play_button = tk.Checkbutton(
            self,
            text="Play Original Spectrogram",
            variable=self.play_spectrogram,
            command=self.toggle_spectrogram_playback,
        )
        play_button.pack(fill="x", side="left", padx=10, pady=10)

        self.play_painted = tk.BooleanVar(value=False)
        play_painted = tk.Checkbutton(
            self,
            text="Play Original Spectrogram",
            variable=self.play_painted,
            command=self.toggle_painted_playback,
        )
        play_painted.pack(fill="x", side="left", padx=10, pady=10)

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
            from_=0.5,
            to=5,
            resolution=0.25,
            orient="horizontal",
            variable=self.radius_tk,
        )
        radius_slider.pack(fill="x", padx=10, pady=10)

    def schedule_next(self, scheduled_func):
        self.paint_job = self.after(self.refresh_rate, scheduled_func)

    def pause_refresh(self, value):
        button = value.button
        if button == 1 and self.running:
            self.running = False
            if self.paint_job is not None:
                self.after_cancel(self.paint_job)
                self.paint_job = None

    def on_close(self):
        self.running = False
        if self.paint_job is not None:
            self.after_cancel(self.paint_job)
            self.paint_job = None
        self.destroy()

    def refresh_canvas(self):
        if not self.running:
            self.paint_job = None
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
        self.schedule_next(self.refresh_canvas)

    def on_mouse_move(self, value):
        if value.xdata is None or value.ydata is None:
            return
        self.pos = np.array([value.xdata, value.ydata])
        button = value.button
        if button == 1 and not self.running:
            self.running = True
            self.schedule_next(self.refresh_canvas)

    def update_spectrogram(self, spectrogram, time, frequencies, tick_count=5):
        im = self.ax.imshow(
            spectrogram,
            origin="lower",
            cmap="hsv",
            aspect="auto",
            extent=[0, 1, 0, 1],
            alpha=0.5,
            zorder=0,
        )

        frequency_samples = np.linspace(0, 1, len(frequencies))
        time_samples = np.linspace(0, 1, len(time))

        self.ax.set_yticks(frequency_samples[:: len(frequencies) // tick_count])
        self.ax.set_yticklabels(
            [f"{f:.1f}Hz" for f in frequencies[:: len(frequencies) // tick_count]]
        )

        self.ax.set_xticks(time_samples[:: len(time) // tick_count])
        self.ax.set_xticklabels([f"{t:.2f}s" for t in time[:: len(time) // tick_count]])

        self.imshow_underlays.append(im)

    def update_spectrogram_visibility(self):
        for im in self.imshow_underlays:
            im.set_alpha(0.5 if not self.hide_spectrogram.get() else 0.0)
        self.canvas.draw_idle()

    def toggle_spectrogram_playback(self):
        pass

    def toggle_painted_playback(self):
        pass
