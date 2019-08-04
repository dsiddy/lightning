import json
import pathlib
import unittest

import cv2 as cv
import natsort

from lightning import (
    DATA_DIR_NAME,
    RANGE,
)
from lightning.utilities import (
    copy_bright_frames,
    get_bright_frames,
    load_frames_from_video_file,
    show_histogram,
    write_video,
)


class RunUtilities(unittest.TestCase):
    def run_copy_bright_frames(self):
        copy_bright_frames()

    def run_get_bright_frames(self):
        bright_frames = []

        frame_file_path_list = list(
            pathlib.Path(f'/Users/dsiddy/Desktop/lightning/data/{DATA_DIR_NAME}').glob('*.jpg')
        )

        mask = cv.imread(
            '/Users/dsiddy/Desktop/lightning/data/mask.bmp',
            0,
        )

        for frame_file_path in frame_file_path_list:
            bright_frames.extend(
                get_bright_frames(
                    frames=[
                        (
                            str(frame_file_path.stem),
                            cv.imread(
                                str(frame_file_path)
                            ),
                        )
                    ],
                    mask=mask,
                    lower_threshold=0.001,
                    # upper_threshold=0.000056,
                    bin_range=range(
                        250,
                        RANGE,
                    ),
                )
            )

        with open('/Users/dsiddy/Desktop/lightning/data/bright_frames-test.json', 'w') as bright_frames_file:
            json.dump(
                bright_frames,
                bright_frames_file,
                indent=2,
            )

    def run_load_frames_from_video_file(self):
        load_frames_from_video_file('/Users/dsiddy/Desktop/lightning/data/IMG_0617.MOV')

    def run_show_histogram(self):
        show_histogram('/Users/dsiddy/Desktop/lightning/data/test/1366.jpg')
        show_histogram('/Users/dsiddy/Desktop/lightning/data/test/1365.jpg')

    def run_write_video(self):
        write_video(
            [
                cv.imread(
                    str(frame_file_path)
                ) for frame_file_path in natsort.natsorted(
                    [
                        str(path.resolve())
                        for path in pathlib.Path('/Users/dsiddy/Desktop/lightning/data/bright_frames').glob('*.jpg')
                    ]
                )
            ],
            '/Users/dsiddy/Desktop/lightning/data/flashes_fast.mp4',
        )


class TestUtilities(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
