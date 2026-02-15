from dataclasses import dataclass, field
import numpy as np
from scipy.signal import istft


@dataclass
class SpectropaintPainter:
    image_width: int = 256
    image_height: int = 256
    panel_height: int = 120
    paint_intensity: float = 10.0
    paint_radius: float = 3.0
    hide_spectrogram: bool = False
    play_spectrogram: bool = False
    play_painted: bool = False
    is_recording: bool = False
    spectrogram_duration: float = 1.0
    spectrogram_time: np.ndarray | None = None
    spectrogram_frequencies: np.ndarray | None = None
    sample_rate: int = 44100
    audio_data_original: np.ndarray | None = None
    spectrogram_playback_pos: float = 0.0
    painted_playback_pos: float = 0.0

    image_data: np.ndarray = field(
        default_factory=lambda: np.zeros((256, 256), dtype=np.uint8)
    )
    spectrogram_rgb: np.ndarray = field(
        default_factory=lambda: np.zeros((256, 256, 3), dtype=np.uint8)
    )

    def clear_paint(self) -> None:
        self.image_data.fill(0)

    def set_intensity(self, value: float) -> None:
        self.paint_intensity = float(np.clip(value, 1.0, 40.0))

    def set_radius(self, value: float) -> None:
        self.paint_radius = float(np.clip(value, 1.0, 20.0))

    def set_audio_data(self, sample_rate: int, audio_data: np.ndarray) -> None:
        audio = np.array(audio_data, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.ndim == 2:
            audio = audio.T

        max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        if max_abs > 1.0:
            audio = audio / 32768.0

        self.sample_rate = int(sample_rate)
        self.audio_data_original = audio
        if audio.shape[1] > 0 and self.sample_rate > 0:
            self.spectrogram_duration = audio.shape[1] / self.sample_rate
        else:
            self.spectrogram_duration = 1.0

    def set_spectrogram(
        self,
        spectrogram: np.ndarray,
        time: np.ndarray,
        frequencies: np.ndarray,
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

        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([255.0, 105.0, 180.0], dtype=np.float32)
        rgb = low + spec_norm[..., None] * (high - low)
        self.spectrogram_rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

        self.spectrogram_time = np.array(time)
        self.spectrogram_frequencies = np.array(frequencies)
        if len(self.spectrogram_time) > 0:
            self.spectrogram_duration = float(self.spectrogram_time[-1])

    def synthesize_from_paint(self) -> np.ndarray | None:
        if self.audio_data_original is None:
            return None

        try:
            painted_magnitudes = self.image_data.astype(np.float32) / 255.0
            _, reconstructed = istft(
                painted_magnitudes,
                fs=self.sample_rate,
                nperseg=1024,
                noverlap=512,
                window="hann",
            )
            reconstructed = np.atleast_2d(reconstructed)
            if reconstructed.ndim == 1:
                reconstructed = reconstructed[np.newaxis, :]

            max_val = (
                float(np.max(np.abs(reconstructed))) if reconstructed.size else 0.0
            )
            if max_val > 0.0:
                reconstructed = reconstructed / max_val
            return reconstructed.astype(np.float32)
        except Exception:
            return self.audio_data_original

    def paint_at_pixel(self, ix: int, iy: int) -> None:
        radius = max(int(self.paint_radius), 1)
        x0 = max(ix - radius, 0)
        x1 = min(ix + radius + 1, self.image_width)
        y0 = max(iy - radius, 0)
        y1 = min(iy + radius + 1, self.image_height)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - iy) ** 2 + (xx - ix) ** 2 <= radius**2
        region = self.image_data[y0:y1, x0:x1]
        region[mask] = np.minimum(region[mask] + self.paint_intensity / 2, 255)
