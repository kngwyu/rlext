import atexit
from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Save video using OpenCV"""

    def __init__(
        self,
        path: Path,
        fps: float = 20.0,
        image_shape: str = "HWC",
        fourcc: str = "XVID",
        rgb: bool = True,
        _close_at_exit: bool = True,
    ) -> None:
        self._path = path
        self._writer = None
        self._fps = fps
        self._w_index = image_shape.find("W")
        self._h_index = image_shape.find("H")
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        if self._w_index < 0 or self._h_index < 0:
            raise ValueError(f"Invalid shape: {image_shape}")
        c_index = image_shape.find("C")
        if c_index < 0:
            transpose = self._h_index, self._w_index
        else:
            transpose = self._h_index, self._w_index, c_index

        def convert(image: np.ndarray) -> np.ndarray:
            transposed = np.transpose(image, transpose)
            if rgb:
                return cv2.cvtColor(transposed, cv2.COLOR_RGB2BGR)
            else:
                return transposed

        self._convert = convert
        if _close_at_exit:
            atexit.register(self.close)

    def _initialize_writer(self, image: np.ndarray) -> None:
        shape = image.shape
        h, w = shape[self._h_index], shape[self._w_index]
        color = len(shape) == 3
        self._writer = cv2.VideoWriter(
            self._path.as_posix(),
            self._fourcc,
            self._fps,
            (w, h),
            color,
        )

    def append(self, image: np.ndarray) -> None:
        if self._writer is None:
            self._initialize_writer(image)

        self._writer.write(self._convert(image))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


class SubProcessVideoWriter:
    def __init__(
        self,
        path: Path,
        fps: float = 20.0,
        image_shape: str = "HWC",
        fourcc: str = "XVID",
        rgb: bool = True,
    ) -> None:
        import multiprocessing as mp
        import traceback

        class Worker(mp.Process):
            def __init__(self, pipe: "Connection") -> None:
                super().__init__()
                self._writer = VideoWriter(
                    path=path,
                    fps=fps,
                    image_shape=image_shape,
                    fourcc=fourcc,
                    rgb=rgb,
                    _close_at_exit=False,
                )
                self._pipe = pipe

            def run(self) -> None:
                try:
                    while True:
                        image = self._pipe.recv()
                        if image is None:
                            self._writer.close()
                            self._pipe.close()
                            break
                        else:
                            self._writer.append(image)
                except Exception as e:
                    traceback.print_exc()
                    print("Exception occured in videowriter process")
                    raise e

        self._pipe, worker_pipe = mp.Pipe()
        self._worker = Worker(worker_pipe)
        self._worker.start()
        atexit.register(self.close)

    def append(self, image: np.ndarray) -> None:
        self._pipe.send(image)

    def close(self) -> None:
        self._pipe.send(None)
        self._pipe.close()
