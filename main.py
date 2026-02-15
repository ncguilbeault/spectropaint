import spectropaint as sp
from spectropaint.window import Window
from spectropaint.spectrogram import Spectrogram


def main():

    window = Window()
    spectrogram = Spectrogram(window, "assets/Bach_prelude_C_major.wav")
    window.run()


if __name__ == "__main__":
    main()
