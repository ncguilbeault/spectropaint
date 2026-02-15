import threading
import time

import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None


class SpectropaintAudio:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._record_thread: threading.Thread | None = None
        self._record_stop_event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._record_thread is not None and self._record_thread.is_alive()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=0.2)

        self.stop_recording()

    def play(
        self,
        audio: np.ndarray | None,
        sample_rate: int,
        *,
        is_painted: bool,
        on_progress,
        on_finished,
        on_error,
    ) -> bool:
        if audio is None:
            return False
        if pyaudio is None:
            on_error("PyAudio not available")
            return False

        self.stop()
        self._stop_event.clear()

        thread = threading.Thread(
            target=self._run_playback,
            args=(
                audio,
                int(sample_rate),
                is_painted,
                on_progress,
                on_finished,
                on_error,
            ),
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return True

    def start_recording(
        self,
        *,
        sample_rate: int,
        channels: int = 1,
        chunk_size: int = 1024,
        on_finished,
        on_error,
    ) -> bool:
        if pyaudio is None:
            on_error("PyAudio not available")
            return False

        self.stop()
        self._record_stop_event.clear()

        thread = threading.Thread(
            target=self._run_recording,
            args=(
                int(sample_rate),
                int(channels),
                int(chunk_size),
                on_finished,
                on_error,
            ),
            daemon=True,
        )
        with self._lock:
            self._record_thread = thread
        thread.start()
        return True

    def stop_recording(self) -> None:
        with self._lock:
            self._record_stop_event.set()
            thread = self._record_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.5)

    def _run_playback(
        self,
        audio: np.ndarray,
        sample_rate: int,
        is_painted: bool,
        on_progress,
        on_finished,
        on_error,
    ) -> None:
        try:
            audio = np.array(audio, dtype=np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            channels, samples = audio.shape
            if samples == 0:
                on_finished(is_painted)
                return

            player = pyaudio.PyAudio()
            stream = player.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                output=True,
            )

            chunk_size = 4096
            start_time = time.time()
            duration = samples / max(sample_rate, 1)

            for i in range(0, samples, chunk_size):
                if self._stop_event.is_set():
                    break

                chunk = audio[:, i : i + chunk_size].T
                stream.write(chunk.astype(np.float32).tobytes())

                elapsed = time.time() - start_time
                pos = min(elapsed / max(duration, 1e-6), 1.0)
                on_progress(is_painted, pos)

            stream.stop_stream()
            stream.close()
            player.terminate()
            on_finished(is_painted)
        except Exception as exc:
            on_error(str(exc))

    def _run_recording(
        self,
        sample_rate: int,
        channels: int,
        chunk_size: int,
        on_finished,
        on_error,
    ) -> None:
        recorder = None
        stream = None
        try:
            recorder = pyaudio.PyAudio()
            stream = recorder.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )

            chunks: list[np.ndarray] = []
            while not self._record_stop_event.is_set():
                data = stream.read(chunk_size, exception_on_overflow=False)
                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    pcm = pcm.reshape(-1, channels).mean(axis=1)
                chunks.append(pcm)

            if chunks:
                audio = np.concatenate(chunks).astype(np.float32)
            else:
                audio = np.zeros((0,), dtype=np.float32)

            on_finished(audio[np.newaxis, :], sample_rate)
        except Exception as exc:
            on_error(str(exc))
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if recorder is not None:
                recorder.terminate()
