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
