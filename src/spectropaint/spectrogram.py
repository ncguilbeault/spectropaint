import numpy as np
import scipy.signal as signal
from scipy.io import wavfile


class Spectrogram:

    def __init__(self, window, wav_file):
        self.window = window
        self.sample_rate, self.audio_data = wavfile.read(wav_file)

        self.frequencies, self.time_segments, self.spectrogram = signal.spectrogram(
            self.audio_data.T,
            fs=self.sample_rate,
            window="hann",
            nperseg=1024,
            noverlap=512,
            scaling="density",
            mode="magnitude",
        )

        self.window.update_spectrogram(
            self.spectrogram[0],
            self.time_segments,
            self.frequencies,
        )
        self.window.set_audio_data(self.sample_rate, self.audio_data)
