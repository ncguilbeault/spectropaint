class SpectropaintController:
    def __init__(self, painter, view, audio, spectrogram):
        self.painter = painter
        self.view = view
        self.audio = audio
        self.spectrogram = spectrogram

        self.view.bind_controller(self)
        self.view.render(self.painter)

    def load_audio_file(self, wav_file: str) -> None:
        sample_rate, audio_data, spec, time, frequencies = self.spectrogram.load(
            wav_file
        )
        self.painter.set_audio_data(sample_rate, audio_data)
        self.painter.set_spectrogram(spec, time, frequencies)
        self.view.render(self.painter)

    def on_toggle_play_original(self):
        if self.painter.is_recording:
            self.view.render(self.painter)
            return

        next_value = not self.painter.play_spectrogram
        self.painter.play_spectrogram = next_value

        if next_value:
            self.painter.play_painted = False
            self.painter.spectrogram_playback_pos = 0.0
            ok = self.audio.play(
                self.painter.audio_data_original,
                self.painter.sample_rate,
                is_painted=False,
                on_progress=self._on_audio_progress,
                on_finished=self._on_audio_finished,
                on_error=self._on_audio_error,
            )
            if not ok:
                self.painter.play_spectrogram = False
        else:
            self.audio.stop()
        self.view.render(self.painter)

    def on_toggle_play_painted(self):
        if self.painter.is_recording:
            self.view.render(self.painter)
            return

        next_value = not self.painter.play_painted
        self.painter.play_painted = next_value

        if next_value:
            self.painter.play_spectrogram = False
            self.painter.painted_playback_pos = 0.0
            painted_audio = self.painter.synthesize_from_paint()
            ok = self.audio.play(
                painted_audio,
                self.painter.sample_rate,
                is_painted=True,
                on_progress=self._on_audio_progress,
                on_finished=self._on_audio_finished,
                on_error=self._on_audio_error,
            )
            if not ok:
                self.painter.play_painted = False
        else:
            self.audio.stop()
        self.view.render(self.painter)

    def on_set_intensity(self, value: float):
        self.painter.set_intensity(value)
        self.view.render(self.painter)

    def on_set_radius(self, value: float):
        self.painter.set_radius(value)
        self.view.render(self.painter)

    def on_toggle_hide(self):
        self.painter.hide_spectrogram = not self.painter.hide_spectrogram
        self.view.render(self.painter)

    def on_clear(self):
        self.painter.clear_paint()
        self.view.render(self.painter)

    def on_toggle_recording(self) -> None:
        if self.painter.is_recording:
            self.audio.stop_recording()
            self.painter.is_recording = False
            self.view.render(self.painter)
            return

        self.audio.stop()
        self.painter.play_spectrogram = False
        self.painter.play_painted = False
        self.painter.spectrogram_playback_pos = 0.0
        self.painter.painted_playback_pos = 0.0

        ok = self.audio.start_recording(
            sample_rate=self.painter.sample_rate,
            channels=1,
            on_finished=self._on_recording_finished,
            on_error=self._on_recording_error,
        )
        self.painter.is_recording = ok
        self.view.render(self.painter)

    def on_paint(self, x: int, y: int):
        ix, iy = self.view.to_image_coords(x, y)
        if ix is None:
            return
        self.painter.paint_at_pixel(ix, iy)
        self.view.render(self.painter)

    def shutdown(self) -> None:
        self.audio.stop()

    def _on_audio_progress(self, is_painted: bool, pos: float) -> None:
        def apply_update() -> None:
            if is_painted:
                self.painter.painted_playback_pos = pos
            else:
                self.painter.spectrogram_playback_pos = pos
            self.view.render(self.painter)

        self.view.dispatch_to_ui(apply_update)

    def _on_audio_finished(self, is_painted: bool) -> None:
        def apply_update() -> None:
            if is_painted:
                self.painter.play_painted = False
                self.painter.painted_playback_pos = 0.0
            else:
                self.painter.play_spectrogram = False
                self.painter.spectrogram_playback_pos = 0.0
            self.view.render(self.painter)

        self.view.dispatch_to_ui(apply_update)

    def _on_audio_error(self, _message: str) -> None:
        def apply_update() -> None:
            self.painter.play_spectrogram = False
            self.painter.play_painted = False
            self.view.render(self.painter)

        self.view.dispatch_to_ui(apply_update)

    def _on_recording_finished(self, audio_data, sample_rate: int) -> None:
        def apply_update() -> None:
            self.painter.is_recording = False
            self.painter.set_audio_data(sample_rate, audio_data)
            spec, time, frequencies = self.spectrogram.from_audio(
                audio_data, sample_rate
            )
            self.painter.set_spectrogram(spec, time, frequencies)
            self.painter.clear_paint()
            self.view.render(self.painter)

        self.view.dispatch_to_ui(apply_update)

    def _on_recording_error(self, _message: str) -> None:
        def apply_update() -> None:
            self.painter.is_recording = False
            self.view.render(self.painter)

        self.view.dispatch_to_ui(apply_update)
