from pathlib import Path
import os

from .controller import SpectropaintController
from .painter import SpectropaintPainter
from .spectrogram import SpectropaintSpectrogram
from .audio import SpectropaintAudio
from .view import SpectropaintView


def main(audio_file: str | None = None) -> None:
    painter = SpectropaintPainter()
    view = SpectropaintView()
    audio = SpectropaintAudio()
    spectrogram = SpectropaintSpectrogram()

    controller = SpectropaintController(painter, view, audio, spectrogram)

    if audio_file is not None:
        controller.load_audio_file(audio_file)

    view.run()


def default_audio_path() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    default_file = str(repo_root / "assets" / "Bach_prelude_C_major.wav")
    if os.path.exists(default_file):
        return default_file
    else:
        None
