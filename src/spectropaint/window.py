import threading
import time

import numpy as np
import pyglet
import pyglet.shapes
from pyglet.window import key, mouse
from scipy.signal import istft

try:
    import pyaudio
except ImportError:
    pyaudio = None


class Window(pyglet.window.Window):

    def __init__(self, image_width: int = 256, image_height: int = 256):
        super().__init__(
            width=900, height=600, caption="Spectrogram Painter", resizable=True
        )

        self.image_width = image_width
        self.image_height = image_height
        self.panel_height = 120

        self.image_data = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint8
        )

        self.hide_spectrogram = False
        self.play_spectrogram = False
        self.play_painted = False

        self.paint_intensity = 10.0
        self.paint_radius = 3.0

        self.spectrogram_time = None
        self.spectrogram_frequencies = None
        self.spectrogram_duration = 1.0
        self.axis_tick_count = 5

        # Single-color spectrum (low -> high). Change these two colors to taste.
        self.colormap_low = np.array([0, 0, 0], dtype=np.float32)
        self.colormap_high = np.array([255, 105, 180], dtype=np.float32)  # pink

        self.spectrogram_playback_pos = 0.0
        self.painted_playback_pos = 0.0
        self._last_update = time.perf_counter()

        # Audio playback
        self.sample_rate = 44100
        self.audio_data_original = None
        self._audio_playback_thread = None
        self._audio_stop_flag = False
        self._audio_lock = threading.Lock()
        self._playback_start_time = None
        self._spectrogram_playback_thread = None
        self._painted_playback_thread = None

        self._spectrogram_rgb = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        self._composite_rgba = np.zeros(
            (self.image_height, self.image_width, 4), dtype=np.uint8
        )

        self._composite_dirty = True

        self._init_render_objects()
        self._init_ui()

        pyglet.clock.schedule_interval(self.update, 1 / 60)

    def _init_render_objects(self) -> None:
        self._base_image = pyglet.image.ImageData(
            self.image_width,
            self.image_height,
            "RGBA",
            self._composite_rgba.tobytes(),
            pitch=self.image_width * 4,
        )
        self._texture = self._base_image.get_texture()
        self._sprite = pyglet.sprite.Sprite(self._texture, x=0, y=0)
        self._sprite.y = self.panel_height
        self._sprite.scale_x = self.width / max(self.image_width, 1)
        self._sprite.scale_y = (self.height - self.panel_height) / max(
            self.image_height, 1
        )

        self._line_batch = pyglet.graphics.Batch()
        self._line_spec = pyglet.shapes.Line(
            0,
            self.panel_height,
            0,
            self.height,
            thickness=2,
            color=(255, 0, 0),
            batch=self._line_batch,
        )
        self._line_paint = pyglet.shapes.Line(
            0,
            self.panel_height,
            0,
            self.height,
            thickness=2,
            color=(0, 255, 0),
            batch=self._line_batch,
        )

        self._axis_batch = pyglet.graphics.Batch()
        self._axis_labels: list[pyglet.text.Label] = []

    def _init_ui(self) -> None:
        self._ui_batch = pyglet.graphics.Batch()
        self._ui_background = pyglet.shapes.Rectangle(
            0,
            0,
            self.width,
            self.panel_height,
            color=(30, 30, 30),
            batch=self._ui_batch,
        )

        self._ui_labels = []
        self._ui_controls = {}
        self._active_slider = None

        self._create_checkbox("hide", "Hide Spectrogram", self.hide_spectrogram)
        self._create_checkbox(
            "play_spec", "Play Original Spectrogram", self.play_spectrogram
        )
        self._create_checkbox("play_paint", "Play Painted", self.play_painted)

        self._create_slider("intensity", "Intensity", 1.0, 40.0, self.paint_intensity)
        self._create_slider("radius", "Radius", 1.0, 20.0, self.paint_radius)

        self._layout_ui()

    def _create_checkbox(self, key_name: str, label: str, value: bool) -> None:
        box = pyglet.shapes.Rectangle(
            0, 0, 14, 14, color=(200, 200, 200), batch=self._ui_batch
        )
        text = pyglet.text.Label(
            label,
            x=0,
            y=0,
            anchor_y="center",
            color=(230, 230, 230, 255),
            batch=self._ui_batch,
        )
        self._ui_controls[key_name] = {
            "type": "checkbox",
            "box": box,
            "selected": None,
            "label": text,
            "value": value,
            "bounds": (0, 0, 0, 0),
        }
        self._set_checkbox_selected(key_name)

    def _set_checkbox_selected(self, key_name: str) -> None:
        control = self._ui_controls[key_name]
        if control["value"]:
            control["selected"] = pyglet.shapes.Rectangle(
                control["box"].x + 3,
                control["box"].y + 3,
                8,
                8,
                color=(0, 0, 0),
                batch=self._ui_batch,
            )
        else:
            if control["selected"] is not None:
                control["selected"].delete()
                control["selected"] = None

    def _create_slider(
        self,
        key_name: str,
        label: str,
        min_value: float,
        max_value: float,
        value: float,
    ) -> None:
        track = pyglet.shapes.Line(
            0, 0, 0, 0, thickness=3, color=(200, 200, 200), batch=self._ui_batch
        )
        knob = pyglet.shapes.Circle(
            0, 0, radius=6, color=(240, 240, 240), batch=self._ui_batch
        )
        text = pyglet.text.Label(
            label,
            x=0,
            y=0,
            anchor_y="center",
            color=(230, 230, 230, 255),
            batch=self._ui_batch,
        )
        value_label = pyglet.text.Label(
            f"{value:.1f}",
            x=0,
            y=0,
            anchor_y="center",
            color=(180, 180, 180, 255),
            batch=self._ui_batch,
        )
        self._ui_controls[key_name] = {
            "type": "slider",
            "track": track,
            "knob": knob,
            "label": text,
            "value_label": value_label,
            "min": min_value,
            "max": max_value,
            "value": value,
            "bounds": (0, 0, 0, 0),
        }

    def _layout_ui(self) -> None:
        margin = 12
        row_y = self.panel_height - 30
        x = margin

        for key_name in ("hide", "play_spec", "play_paint"):
            control = self._ui_controls[key_name]
            box = control["box"]
            label = control["label"]

            box.x = x
            box.y = row_y - 7
            label.x = x + 20
            label.y = row_y

            label_width = label.content_width
            total_width = 20 + label_width + 14
            control["bounds"] = (x, row_y - 8, total_width, 16)
            x += total_width + 20

        slider_row = 35
        slider_x = margin
        slider_width = 220

        for key_name in ("intensity", "radius"):
            control = self._ui_controls[key_name]
            label = control["label"]
            track = control["track"]
            knob = control["knob"]
            value_label = control["value_label"]

            label.x = slider_x
            label.y = slider_row + 20
            track.x = slider_x
            track.y = slider_row
            track.x2 = slider_x + slider_width
            track.y2 = slider_row

            knob.y = slider_row
            value_label.y = slider_row + 20

            self._set_slider_value(key_name, control["value"])
            value_label.x = slider_x + slider_width + 50

            control["bounds"] = (
                slider_x,
                slider_row - 8,
                slider_width,
                16,
            )

            slider_x += slider_width + 160

        self._ui_background.width = self.width
        self._ui_background.height = self.panel_height

    def _set_slider_value(self, key_name: str, value: float) -> None:
        control = self._ui_controls[key_name]
        value = float(np.clip(value, control["min"], control["max"]))
        control["value"] = value

        x0, y0, width, _ = control["bounds"]
        if width == 0:
            width = 1
        t = (value - control["min"]) / max(control["max"] - control["min"], 1e-6)
        control["knob"].x = x0 + t * width
        control["value_label"].text = f"{value:.1f}"

        if key_name == "intensity":
            self.paint_intensity = value
        elif key_name == "radius":
            self.paint_radius = value

    def set_audio_data(self, sample_rate: int, audio_data: np.ndarray) -> None:
        """Store audio data for playback."""
        with self._audio_lock:
            self.sample_rate = sample_rate
            self.audio_data_original = np.array(audio_data, dtype=np.float32)
            if self.audio_data_original.ndim == 1:
                self.audio_data_original = self.audio_data_original[np.newaxis, :]
            elif self.audio_data_original.ndim == 2:
                self.audio_data_original = self.audio_data_original.T
            if self.audio_data_original.max() > 1.0:
                self.audio_data_original = self.audio_data_original / 32768.0
            self.spectrogram_duration = len(self.audio_data_original[0]) / sample_rate

    def _play_audio(self, audio: np.ndarray, is_painted: bool = False) -> None:
        """Play audio data through speaker using pyaudio."""
        if pyaudio is None or audio is None:
            print("PyAudio not available or audio is None")
            return

        try:
            audio = np.clip(audio, -1.0, 1.0)
            num_channels = audio.shape[0]

            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=num_channels,
                rate=self.sample_rate,
                output=True,
            )

            chunk_size = 4096
            self._playback_start_time = time.time()

            for i in range(0, audio.shape[1], chunk_size):
                if self._audio_stop_flag:
                    break
                chunk = audio[:, i : i + chunk_size].T
                try:
                    stream.write(chunk.astype(np.float32).tobytes())
                except Exception as e:
                    print(f"Error writing chunk: {e}")
                    break

                elapsed = time.time() - self._playback_start_time
                total_duration = audio.shape[1] / self.sample_rate
                pos = min(elapsed / total_duration, 1.0)
                if is_painted:
                    self.painted_playback_pos = pos
                else:
                    self.spectrogram_playback_pos = pos

            stream.stop_stream()
            stream.close()
            p.terminate()

            if is_painted:
                self.play_painted = False
                self._ui_controls["play_paint"]["value"] = False
                self._set_checkbox_selected("play_paint")
            else:
                self.play_spectrogram = False
                self._ui_controls["play_spec"]["value"] = False
                self._set_checkbox_selected("play_spec")

            self._audio_stop_flag = False
        except Exception as e:
            print(f"Audio playback error: {e}")
            self._audio_stop_flag = False

    def run(self) -> None:
        pyglet.app.run()

    def on_draw(self) -> None:
        self.clear()

        if self._composite_dirty:
            self._rebuild_composite()

        self._sprite.draw()

        self._update_playback_lines()
        self._line_batch.draw()

        self._axis_batch.draw()

        self._ui_batch.draw()

        self.invalid = False

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._sprite.scale_x = width / max(self.image_width, 1)
        self._sprite.scale_y = (height - self.panel_height) / max(self.image_height, 1)
        self._sprite.y = self.panel_height
        self._line_spec.y = self.panel_height
        self._line_spec.y2 = height
        self._line_paint.y = self.panel_height
        self._line_paint.y2 = height

        self._layout_ui()
        self._update_axis_labels()
        self.invalid = True

    def update(self, dt: float) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_update
        self._last_update = now

        if self.play_spectrogram:
            self.spectrogram_playback_pos = (
                self.spectrogram_playback_pos
                + elapsed / max(self.spectrogram_duration, 1e-6)
            ) % 1.0
        if self.play_painted:
            self.painted_playback_pos = (
                self.painted_playback_pos
                + elapsed / max(self.spectrogram_duration, 1e-6)
            ) % 1.0

        self.invalid = True

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == mouse.LEFT:
            if self._handle_ui_click(x, y):
                self.invalid = True
                return
            self._paint_at(x, y)
            self.invalid = True

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        if self._active_slider is not None:
            self._handle_slider_drag(x)
            self.invalid = True
            return
        if buttons & mouse.LEFT:
            self._paint_at(x, y)
            self.invalid = True

    def _paint_at(self, x: int, y: int) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            return

        if y <= self.panel_height:
            return

        plot_height = max(self.height - self.panel_height, 1)
        u = np.clip(x / max(self.width, 1), 0.0, 1.0)
        v = np.clip((y - self.panel_height) / plot_height, 0.0, 1.0)

        ix = int(u * (self.image_width - 1))
        iy = int(v * (self.image_height - 1))
        radius = max(int(self.paint_radius), 1)

        x0 = max(ix - radius, 0)
        x1 = min(ix + radius + 1, self.image_width)
        y0 = max(iy - radius, 0)
        y1 = min(iy + radius + 1, self.image_height)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - iy) ** 2 + (xx - ix) ** 2 <= radius**2

        region = self.image_data[y0:y1, x0:x1]
        region[mask] = np.minimum(region[mask] + self.paint_intensity, 255)

        self._composite_dirty = True
        self.invalid = True

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == key.H:
            self.hide_spectrogram = not self.hide_spectrogram
            self._composite_dirty = True
            self._ui_controls["hide"]["value"] = self.hide_spectrogram
            self._set_checkbox_selected("hide")
        elif symbol == key.O:
            self.play_spectrogram = not self.play_spectrogram
            self._ui_controls["play_spec"]["value"] = self.play_spectrogram
            self._set_checkbox_selected("play_spec")
            if self.play_spectrogram:
                self._audio_stop_flag = True
                if self._audio_playback_thread:
                    self._audio_playback_thread.join(timeout=0.1)
                self._audio_stop_flag = False
                self.spectrogram_playback_pos = 0.0
                self._audio_playback_thread = threading.Thread(
                    target=self._play_audio,
                    args=(self.audio_data_original, False),
                    daemon=True,
                )
                self._audio_playback_thread.start()
            else:
                self._audio_stop_flag = True
        elif symbol == key.P:
            self.play_painted = not self.play_painted
            self._ui_controls["play_paint"]["value"] = self.play_painted
            self._set_checkbox_selected("play_paint")
            if self.play_painted:
                self._audio_stop_flag = True
                if self._audio_playback_thread:
                    self._audio_playback_thread.join(timeout=0.1)
                self._audio_stop_flag = False
                self.painted_playback_pos = 0.0
                synth_audio = self._synthesize_from_paint()
                self._audio_playback_thread = threading.Thread(
                    target=self._play_audio,
                    args=(synth_audio, True),
                    daemon=True,
                )
                self._audio_playback_thread.start()
            else:
                self._audio_stop_flag = True
        elif symbol in (key.EQUAL, key.NUM_ADD):
            self.paint_intensity = min(self.paint_intensity + 1.0, 40.0)
            self._set_slider_value("intensity", self.paint_intensity)
        elif symbol in (key.MINUS, key.UNDERSCORE, key.NUM_SUBTRACT):
            self.paint_intensity = max(self.paint_intensity - 1.0, 1.0)
            self._set_slider_value("intensity", self.paint_intensity)
        elif symbol == key.BRACKETRIGHT:
            self.paint_radius = min(self.paint_radius + 1.0, 20.0)
            self._set_slider_value("radius", self.paint_radius)
        elif symbol == key.BRACKETLEFT:
            self.paint_radius = max(self.paint_radius - 1.0, 1.0)
            self._set_slider_value("radius", self.paint_radius)
        elif symbol == key.C:
            self.image_data.fill(0)
            self._composite_dirty = True
        self.invalid = True

    def update_spectrogram(
        self, spectrogram, time, frequencies, tick_count: int = 5
    ) -> None:
        spec = np.array(spectrogram, dtype=np.float32)
        if spec.ndim != 2:
            raise ValueError("Spectrogram must be 2D")

        spec = np.log1p(spec)
        spec_min = float(spec.min())
        spec_max = float(spec.max())
        spec_norm = (spec - spec_min) / max(spec_max - spec_min, 1e-6)

        height, width = spec_norm.shape
        if (width, height) != (self.image_width, self.image_height):
            self.image_width = width
            self.image_height = height
            self.image_data = np.zeros((height, width), dtype=np.uint8)
            self._spectrogram_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            self._composite_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            self._base_image = pyglet.image.ImageData(
                width,
                height,
                "RGBA",
                self._composite_rgba.tobytes(),
                pitch=width * 4,
            )
            self._texture = self._base_image.get_texture()
            self._sprite = pyglet.sprite.Sprite(self._texture, x=0, y=0)
            self._sprite.y = self.panel_height
            self._sprite.scale_x = self.width / max(self.image_width, 1)
            self._sprite.scale_y = (self.height - self.panel_height) / max(
                self.image_height, 1
            )

        self._spectrogram_rgb = self._apply_colormap(spec_norm)
        self._composite_dirty = True
        self.invalid = True

        self.spectrogram_time = time
        self.spectrogram_frequencies = frequencies
        if time is not None and len(time) > 0:
            self.spectrogram_duration = float(time[-1])
        else:
            self.spectrogram_duration = 1.0

        self.axis_tick_count = max(int(tick_count), 2)
        self._update_axis_labels()

    def _apply_colormap(self, values: np.ndarray) -> np.ndarray:
        values = np.clip(values, 0.0, 1.0).astype(np.float32)
        rgb = self.colormap_low + values[..., None] * (
            self.colormap_high - self.colormap_low
        )
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def _synthesize_from_paint(self) -> np.ndarray:
        """Synthesize audio from painted spectrogram data using inverse STFT."""
        if self.spectrogram_frequencies is None or self.audio_data_original is None:
            return self.audio_data_original

        try:
            # Normalize painted data to 0-1 range (treat as magnitudes)
            painted_magnitudes = self.image_data.astype(np.float32) / 255.0

            # Use inverse STFT with just the magnitudes (phase is generated automatically)
            nperseg = 1024
            noverlap = 512

            _, reconstructed = istft(
                painted_magnitudes,
                fs=self.sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )

            # Convert to mono format (1, n_samples)
            reconstructed = np.atleast_2d(reconstructed)
            if reconstructed.ndim == 1:
                reconstructed = reconstructed[np.newaxis, :]

            # Normalize
            max_val = np.max(np.abs(reconstructed))
            if max_val > 0:
                reconstructed = reconstructed / max_val

            return reconstructed.astype(np.float32)
        except Exception as e:
            print(f"Error synthesizing audio from paint: {e}")
            return self.audio_data_original

    def _rebuild_composite(self) -> None:
        spec_alpha = 0.0 if self.hide_spectrogram else 0.6
        paint_alpha = 1.0

        if self._spectrogram_rgb.shape[:2] != self.image_data.shape:
            self._spectrogram_rgb = np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.uint8
            )

        spec = self._spectrogram_rgb.astype(np.float32)
        paint = self.image_data.astype(np.float32)

        composite_rgb = np.clip(
            spec * spec_alpha + paint[..., None] * paint_alpha, 0.0, 255.0
        ).astype(np.uint8)

        alpha = np.full((self.image_height, self.image_width, 1), 255, dtype=np.uint8)
        self._composite_rgba = np.concatenate([composite_rgb, alpha], axis=-1)

        flipped = self._composite_rgba
        self._base_image.set_data("RGBA", self.image_width * 4, flipped.tobytes())
        self._texture.blit_into(self._base_image, 0, 0, 0)
        self._composite_dirty = False

    def _update_playback_lines(self) -> None:
        x_spec = self.width * float(self.spectrogram_playback_pos)
        x_paint = self.width * float(self.painted_playback_pos)

        self._line_spec.x = x_spec
        self._line_spec.x2 = x_spec
        self._line_spec.opacity = 255 if self.play_spectrogram else 0

        self._line_paint.x = x_paint
        self._line_paint.x2 = x_paint
        self._line_paint.opacity = 255 if self.play_painted else 0

    def _update_axis_labels(self) -> None:
        for label in self._axis_labels:
            label.delete()
        self._axis_labels = []

        if (
            self.spectrogram_time is None
            or self.spectrogram_frequencies is None
            or len(self.spectrogram_time) == 0
            or len(self.spectrogram_frequencies) == 0
        ):
            return

        tick_count = max(self.axis_tick_count, 2)
        plot_height = max(self.height - self.panel_height, 1)

        time_ticks = np.linspace(0, len(self.spectrogram_time) - 1, tick_count).astype(
            int
        )
        freq_ticks = np.linspace(
            0, len(self.spectrogram_frequencies) - 1, tick_count
        ).astype(int)

        for idx in time_ticks:
            t = float(self.spectrogram_time[idx])
            x = (idx / max(len(self.spectrogram_time) - 1, 1)) * self.width
            label = pyglet.text.Label(
                f"{t:.2f}s",
                x=x,
                y=self.panel_height + 4,
                anchor_x="center",
                anchor_y="bottom",
                color=(220, 220, 220, 255),
                batch=self._axis_batch,
            )
            self._axis_labels.append(label)

        for idx in freq_ticks:
            f = float(self.spectrogram_frequencies[idx])
            y = (
                self.panel_height
                + (idx / max(len(self.spectrogram_frequencies) - 1, 1)) * plot_height
            )
            label = pyglet.text.Label(
                f"{f:.1f}Hz",
                x=6,
                y=y,
                anchor_x="left",
                anchor_y="center",
                color=(220, 220, 220, 255),
                batch=self._axis_batch,
            )
            self._axis_labels.append(label)

    def _handle_ui_click(self, x: int, y: int) -> bool:
        if y > self.panel_height:
            return False

        for key_name in ("hide", "play_spec", "play_paint"):
            control = self._ui_controls[key_name]
            bx, by, bw, bh = control["bounds"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                control["value"] = not control["value"]
                self._set_checkbox_selected(key_name)
                if key_name == "hide":
                    self.hide_spectrogram = control["value"]
                    self._composite_dirty = True
                elif key_name == "play_spec":
                    self.play_spectrogram = control["value"]
                    if self.play_spectrogram and self.audio_data_original is not None:
                        self._audio_stop_flag = True
                        if self._audio_playback_thread:
                            self._audio_playback_thread.join(timeout=0.1)
                        self._audio_stop_flag = False
                        self.spectrogram_playback_pos = 0.0
                        self._audio_playback_thread = threading.Thread(
                            target=self._play_audio,
                            args=(self.audio_data_original, False),
                            daemon=True,
                        )
                        self._audio_playback_thread.start()
                    else:
                        self._audio_stop_flag = True
                elif key_name == "play_paint":
                    self.play_painted = control["value"]
                    if self.play_painted and self.audio_data_original is not None:
                        self._audio_stop_flag = True
                        if self._audio_playback_thread:
                            self._audio_playback_thread.join(timeout=0.1)
                        self._audio_stop_flag = False
                        self.painted_playback_pos = 0.0
                        synth_audio = self._synthesize_from_paint()
                        self._audio_playback_thread = threading.Thread(
                            target=self._play_audio,
                            args=(synth_audio, True),
                            daemon=True,
                        )
                        self._audio_playback_thread.start()
                    else:
                        self._audio_stop_flag = True
                return True

        for key_name in ("intensity", "radius"):
            control = self._ui_controls[key_name]
            bx, by, bw, bh = control["bounds"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._active_slider = key_name
                self._handle_slider_drag(x)
                return True

        return False

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == mouse.LEFT:
            self._active_slider = None

    def _handle_slider_drag(self, x: int) -> None:
        key_name = self._active_slider
        if key_name is None:
            return
        control = self._ui_controls[key_name]
        bx, _, bw, _ = control["bounds"]
        t = (x - bx) / max(bw, 1)
        value = control["min"] + t * (control["max"] - control["min"])
        self._set_slider_value(key_name, value)
        self._composite_dirty = (
            True if key_name == "intensity" else self._composite_dirty
        )
