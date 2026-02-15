import numpy as np
import scipy.signal as signal
from scipy.io import wavfile


class SpectropaintSpectrogram:
    def load(
        self, wav_file: str
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_rate, audio_data = wavfile.read(wav_file)

        working = np.array(audio_data)
        if working.ndim == 2:
            mono = working.mean(axis=1)
        else:
            mono = working

        frequencies, time_segments, spectrogram = signal.spectrogram(
            mono,
            fs=sample_rate,
            window="hann",
            nperseg=1024,
            noverlap=512,
            scaling="density",
            mode="magnitude",
        )

        return sample_rate, audio_data, spectrogram, time_segments, frequencies

    def from_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        working = np.array(audio_data)

        if working.ndim == 2:
            # Accept both (samples, channels) and (channels, samples).
            if working.shape[0] <= 8 and working.shape[1] > working.shape[0]:
                mono = working.mean(axis=0)
            else:
                mono = working.mean(axis=1)
        else:
            mono = working

        frequencies, time_segments, spectrogram = signal.spectrogram(
            mono,
            fs=sample_rate,
            window="hann",
            nperseg=1024,
            noverlap=512,
            scaling="density",
            mode="magnitude",
        )

        return spectrogram, time_segments, frequencies
