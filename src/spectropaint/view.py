import numpy as np
import pyglet
import pyglet.shapes
from pyglet.window import key, mouse


class SpectropaintView(pyglet.window.Window):

    def __init__(self) -> None:
        super().__init__(width=900, height=600, caption="SpectroPaint", resizable=True)

        self.panel_height = 120
        self.axis_tick_count = 5

        self.image_width = 256
        self.image_height = 256

        self.hide_spectrogram = False
        self.play_spectrogram = False
        self.play_painted = False
        self.is_recording = False
        self.paint_intensity = 10.0
        self.paint_radius = 3.0

        self.spectrogram_playback_pos = 0.0
        self.painted_playback_pos = 0.0
        self.spectrogram_time = None
        self.spectrogram_frequencies = None

        self._controller = None
        self._active_slider = None
        self._composite_dirty = True
        self._hover_x = 0
        self._hover_y = 0
        self._hover_visible = False

        self._image_data = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint8
        )
        self._spectrogram_rgb = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        self._composite_rgba = np.zeros(
            (self.image_height, self.image_width, 4), dtype=np.uint8
        )

        self._init_render_objects()
        self._init_ui()

        self._hover_circle = pyglet.shapes.Circle(
            0,
            0,
            radius=8,
            color=(255, 0, 0),
        )
        self._hover_circle.opacity = 120

        pyglet.clock.schedule_interval(self.update, 1 / 60)

    def bind_controller(self, controller) -> None:
        self._controller = controller

    def dispatch_to_ui(self, callback) -> None:
        pyglet.clock.schedule_once(lambda _dt: callback(), 0.0)

    def run(self) -> None:
        pyglet.app.run()

    def invalidate(self) -> None:
        self._composite_dirty = True
        self.invalid = True

    def render(self, model) -> None:
        resized = (self.image_width, self.image_height) != (
            model.image_width,
            model.image_height,
        )

        self.hide_spectrogram = model.hide_spectrogram
        self.play_spectrogram = model.play_spectrogram
        self.play_painted = model.play_painted
        self.is_recording = model.is_recording
        self.paint_intensity = model.paint_intensity
        self.paint_radius = model.paint_radius
        self.spectrogram_playback_pos = model.spectrogram_playback_pos
        self.painted_playback_pos = model.painted_playback_pos
        self.spectrogram_time = model.spectrogram_time
        self.spectrogram_frequencies = model.spectrogram_frequencies

        self.image_width = model.image_width
        self.image_height = model.image_height
        self._image_data = model.image_data
        self._spectrogram_rgb = model.spectrogram_rgb

        if resized:
            self._composite_rgba = np.zeros(
                (self.image_height, self.image_width, 4), dtype=np.uint8
            )
            self._init_render_objects()
            self._layout_ui()

        self._set_checkbox_value("hide", self.hide_spectrogram)
        self._set_checkbox_value("play_spec", self.play_spectrogram)
        self._set_checkbox_value("play_paint", self.play_painted)
        self._set_checkbox_value("record", self.is_recording)
        self._set_slider_value("intensity", self.paint_intensity)
        self._set_slider_value("radius", self.paint_radius)

        self._update_axis_labels()
        self.invalidate()

    def to_image_coords(self, x: int, y: int):
        if self.image_width <= 0 or self.image_height <= 0 or y <= self.panel_height:
            return None, None

        plot_height = max(self.height - self.panel_height, 1)
        u = np.clip(x / max(self.width, 1), 0.0, 1.0)
        v = np.clip((y - self.panel_height) / plot_height, 0.0, 1.0)
        ix = int(u * (self.image_width - 1))
        iy = int(v * (self.image_height - 1))
        return ix, iy

    def update(self, _dt: float) -> None:
        self.invalid = True

    def on_draw(self) -> None:
        self.clear()
        if self._composite_dirty:
            self._rebuild_composite()
        self._sprite.draw()
        self._draw_hover_brush_preview()
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

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button != mouse.LEFT:
            return
        if self._handle_ui_click(x, y):
            return
        self._set_hover_position(x, y)
        if self._controller is not None:
            self._controller.on_paint(x, y)

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        self._set_hover_position(x, y)
        if self._active_slider is not None:
            self._handle_slider_drag(x)
            return
        if buttons & mouse.LEFT and self._controller is not None:
            self._controller.on_paint(x, y)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        if button == mouse.LEFT:
            self._active_slider = None

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self._set_hover_position(x, y)

    def on_mouse_leave(self, x: int, y: int) -> None:
        self._hover_visible = False
        self.invalid = True

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if self._controller is None:
            return

        if symbol == key.H:
            self._controller.on_toggle_hide()
        elif symbol == key.O:
            self._controller.on_toggle_play_original()
        elif symbol == key.P:
            self._controller.on_toggle_play_painted()
        elif symbol in (key.EQUAL, key.NUM_ADD):
            self._controller.on_set_intensity(self.paint_intensity + 1.0)
        elif symbol in (key.MINUS, key.UNDERSCORE, key.NUM_SUBTRACT):
            self._controller.on_set_intensity(self.paint_intensity - 1.0)
        elif symbol == key.BRACKETRIGHT:
            self._controller.on_set_radius(self.paint_radius + 1.0)
        elif symbol == key.BRACKETLEFT:
            self._controller.on_set_radius(self.paint_radius - 1.0)
        elif symbol == key.C:
            self._controller.on_clear()
        elif symbol == key.R:
            self._controller.on_toggle_recording()

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

        self._ui_controls = {}
        self._create_checkbox("hide", "Hide Spectrogram", self.hide_spectrogram)
        self._create_checkbox(
            "play_spec", "Play Original Spectrogram", self.play_spectrogram
        )
        self._create_checkbox("play_paint", "Play Painted", self.play_painted)
        self._create_checkbox("record", "Record Mic", self.is_recording)
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
            "box": box,
            "selected": None,
            "label": text,
            "value": value,
            "bounds": (0, 0, 0, 0),
        }
        self._set_checkbox_selected(key_name)

    def _set_checkbox_value(self, key_name: str, value: bool) -> None:
        self._ui_controls[key_name]["value"] = bool(value)
        self._set_checkbox_selected(key_name)

    def _set_checkbox_selected(self, key_name: str) -> None:
        c = self._ui_controls[key_name]
        if c["value"]:
            c["selected"] = pyglet.shapes.Rectangle(
                c["box"].x + 3,
                c["box"].y + 3,
                8,
                8,
                color=(0, 0, 0),
                batch=self._ui_batch,
            )
        elif c["selected"] is not None:
            c["selected"].delete()
            c["selected"] = None

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
            "track": track,
            "knob": knob,
            "label": text,
            "value_label": value_label,
            "min": min_value,
            "max": max_value,
            "value": value,
            "bounds": (0, 0, 0, 0),
        }

    def _set_slider_value(self, key_name: str, value: float) -> None:
        c = self._ui_controls[key_name]
        value = float(np.clip(value, c["min"], c["max"]))
        c["value"] = value
        x0, y0, width, height = c["bounds"]
        width = max(width, 1)
        t = (value - c["min"]) / max(c["max"] - c["min"], 1e-6)
        c["knob"].x = x0 + t * width
        c["knob"].y = y0 + (height / 2)
        c["value_label"].text = f"{value:.1f}"

    def _layout_ui(self) -> None:
        margin = 12
        row_y = self.panel_height - 30
        x = margin

        for key_name in ("hide", "play_spec", "play_paint", "record"):
            c = self._ui_controls[key_name]
            c["box"].x = x
            c["box"].y = row_y - 7
            c["label"].x = x + 20
            c["label"].y = row_y
            total_width = 20 + c["label"].content_width + 14
            c["bounds"] = (x, row_y - 8, total_width, 16)
            x += total_width + 20

        slider_row = 35
        slider_x = margin
        slider_width = 220

        for key_name in ("intensity", "radius"):
            c = self._ui_controls[key_name]
            c["label"].x = slider_x
            c["label"].y = slider_row + 20
            c["track"].x = slider_x
            c["track"].y = slider_row
            c["track"].x2 = slider_x + slider_width
            c["track"].y2 = slider_row
            c["knob"].y = slider_row
            c["value_label"].y = slider_row + 20
            c["bounds"] = (slider_x, slider_row - 8, slider_width, 16)
            self._set_slider_value(key_name, c["value"])
            c["value_label"].x = slider_x + slider_width + 50
            slider_x += slider_width + 160

        self._ui_background.width = self.width
        self._ui_background.height = self.panel_height

    def _handle_ui_click(self, x: int, y: int) -> bool:
        if y > self.panel_height or self._controller is None:
            return False

        for key_name in ("hide", "play_spec", "play_paint", "record"):
            c = self._ui_controls[key_name]
            bx, by, bw, bh = c["bounds"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if key_name == "hide":
                    self._controller.on_toggle_hide()
                elif key_name == "play_spec":
                    self._controller.on_toggle_play_original()
                elif key_name == "play_paint":
                    self._controller.on_toggle_play_painted()
                elif key_name == "record":
                    self._controller.on_toggle_recording()
                return True

        for key_name in ("intensity", "radius"):
            c = self._ui_controls[key_name]
            bx, by, bw, bh = c["bounds"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._active_slider = key_name
                self._handle_slider_drag(x)
                return True

        return False

    def _handle_slider_drag(self, x: int) -> None:
        if self._active_slider is None or self._controller is None:
            return
        c = self._ui_controls[self._active_slider]
        bx, _, bw, _ = c["bounds"]
        t = (x - bx) / max(bw, 1)
        value = c["min"] + t * (c["max"] - c["min"])
        if self._active_slider == "intensity":
            self._controller.on_set_intensity(value)
        else:
            self._controller.on_set_radius(value)

    def _set_hover_position(self, x: int, y: int) -> None:
        self._hover_x = x
        self._hover_y = y
        self._hover_visible = y > self.panel_height
        self.invalid = True

    def _draw_hover_brush_preview(self) -> None:
        if not self._hover_visible:
            return

        if self._hover_y <= self.panel_height:
            return

        scale_x = self.width / max(self.image_width, 1)
        scale_y = (self.height - self.panel_height) / max(self.image_height, 1)
        radius_px = max(float(self.paint_radius) * min(scale_x, scale_y), 1.0)
        alpha = int(min((self.paint_intensity + 10) / 40.0, 1) * 180)

        self._hover_circle.x = self._hover_x
        self._hover_circle.y = self._hover_y
        self._hover_circle.radius = radius_px
        self._hover_circle.opacity = alpha
        self._hover_circle.draw()

    def _rebuild_composite(self) -> None:
        spec_alpha = 0.0 if self.hide_spectrogram else 0.6
        spec = self._spectrogram_rgb.astype(np.float32)
        paint = self._image_data.astype(np.float32)
        composite_rgb = np.clip(
            spec * spec_alpha + paint[..., None], 0.0, 255.0
        ).astype(np.uint8)
        alpha = np.full((self.image_height, self.image_width, 1), 255, dtype=np.uint8)
        self._composite_rgba = np.concatenate([composite_rgb, alpha], axis=-1)
        self._base_image.set_data(
            "RGBA", self.image_width * 4, self._composite_rgba.tobytes()
        )
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
            x = (idx / max(len(self.spectrogram_time) - 1, 1)) * (self.width * 0.94) + (
                self.width * 0.03
            )
            self._axis_labels.append(
                pyglet.text.Label(
                    f"{t:.2f}s",
                    x=x,
                    y=self.panel_height + 4,
                    anchor_x="center",
                    anchor_y="bottom",
                    color=(220, 220, 220, 255),
                    batch=self._axis_batch,
                )
            )

        for idx in freq_ticks:
            f = float(self.spectrogram_frequencies[idx])
            y = (
                self.panel_height
                + (idx / max(len(self.spectrogram_frequencies) - 1, 1))
                * (plot_height * 0.98)
                + (plot_height * 0.01)
            )
            self._axis_labels.append(
                pyglet.text.Label(
                    f"{f:.1f}Hz",
                    x=6,
                    y=y,
                    anchor_x="left",
                    anchor_y="center",
                    color=(220, 220, 220, 255),
                    batch=self._axis_batch,
                )
            )
