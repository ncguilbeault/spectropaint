# SpectroPaint

OpenGL-based spectrogram painter. This interactive demo allows you to record audio, visualize the audio as a spectrogram over time, and playback the audio. The demo also allows you to "paint" a spectrogram which can be visualized overtop of the audio recording and can be played back separately.

## Quickstart

Install the `uv` Python environment manager. Then, simply run the following command in a terminal:

```bash
git clone https://github.com/ncguilbeault/spectropaint.git
cd spectropaint/
uv run main.py
```

## Controls

- Left mouse drag: paint
- H: toggle spectrogram visibility
- O: toggle spectrogram playback line
- P: toggle painted playback line
- R: start/stop microphone recording (updates spectrogram)
- + / -: brush intensity
- [ / ]: brush radius
- C: clear paint

## Screenshot

![](./assets/screenshot.png)