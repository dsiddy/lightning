import json
import pathlib
import shutil
from typing import (
    List,
    Tuple,
    Union,
)

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from . import (
    DATA_DIR_NAME,
    RANGE,
)


def copy_bright_frames() -> None:
    DATA_BASE_DIR_PATH = pathlib.Path('/Users/dsiddy/Desktop/lightning/data')

    BRIGHT_FRAMES_DIR_PATH = DATA_BASE_DIR_PATH.joinpath('bright_frames')
    BRIGHT_FRAMES_DIR_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    BRIGHT_FRAMES_LIST_FILE_PATH = DATA_BASE_DIR_PATH.joinpath('bright_frames.json')

    with open(BRIGHT_FRAMES_LIST_FILE_PATH) as bright_frames_list_file:
        bright_frames_list = json.load(bright_frames_list_file)

    for bright_frame in bright_frames_list:
        shutil.copyfile(
            DATA_BASE_DIR_PATH.joinpath(f'{DATA_DIR_NAME}/{bright_frame}.jpg'),
            BRIGHT_FRAMES_DIR_PATH.joinpath(f'{bright_frame}.jpg'),
        )


def get_bright_frames(
    frames: List[
        Tuple[
            str,
            np.ndarray,
        ]
    ],
    mask: Union[
        np.ndarray,
        None,
    ] = None,
    num_bins: int = RANGE,
    lower_threshold: float = 0.5,
    upper_threshold: float = 1,
    bin_range: List[int] = range(RANGE // 2, RANGE),
):
    """
    If the proportion of the total pixels falling within a given bin range exceeds `threshold`, add the frame file path to a list.

    Args:
        `frames`: A list of tuples, each of which contains the file path to the frame and the frame data itself.

        `mask`: An `np.ndarray` instance that represents a bit mask, or `None`. (See, *e.g.*, <https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html>.)

        `num_bins`: The number of bins in the histogram.

        `threshold`: The proportion of pixels that needs to be exceeded in order for the frame to be classified as 'bright' for our purposes.

        `bin_range`: The bins over which to range when we count pixels.
    Returns:
        A list of file paths to the frames that satisfy our criteria.
    """
    assert 0 <= lower_threshold <= 1
    assert 0 <= upper_threshold <= 1
    assert lower_threshold < upper_threshold
    assert bin_range[0] >= 0
    assert bin_range[-1] <= num_bins

    bright_frames = set()

    for frame in frames:
        frame_file_path = frame[0]
        frame_data = frame[1]

        num_pixels = frame_data.shape[0] * frame_data.shape[1]

        histogram = cv.calcHist(
            [frame_data],
            [0],
            mask,
            [num_bins],
            [0, num_bins],
        )

        pixel_count = 0

        for index in bin_range:
            pixel_count += histogram[index]

        pixel_proportion = pixel_count / num_pixels

        if (pixel_proportion >= lower_threshold) and (pixel_proportion <= upper_threshold):
            bright_frames.add(frame_file_path)

            # Append nearby frames as well.
            # TODO(dsiddy): This approach won't work insofar as we're passing the frames individually (and with good reason; we don't want to load 30,000 frames into memory all at once, even if we could).
            frame_index = frames.index(frame)

            try:
                bright_frames.add(frames[frame_index - 1][0])
                bright_frames.add(frames[frame_index + 1][0])
            except IndexError:
                pass

    return bright_frames


def get_mean_brightness(
    frame: np.ndarray,
    mask: Union[
        np.ndarray,
        None,
    ] = None,
) -> int:
    """Return the mean brightness of a frame.

    Load the frame, calculate a histogram, and iterate through the bins until half or more of the pixels have been counted.

    Args:
        `frame`: A video data frame.

        `mask`: An `np.ndarray` instance that represents a bit mask, or `None`. (See, *e.g.*, <https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html>.)
    Returns:
        A integer representing the mean brightness of the frame. (Note that this is defined relative to the number of bins in the histogram.)
    """
    try:
        grayscale_frame = cv.cvtColor(
            frame,
            cv.COLOR_RGB2GRAY,
        )
    except Exception as error:
        print(f'Could not convert frame to grayscale. ({error})')

        return False

    num_pixels = frame.shape[0] * frame.shape[1]

    histogram = cv.calcHist(
        [grayscale_frame],
        [0],
        mask,
        [RANGE],
        [0, RANGE],
    )

    pixel_count = 0
    bin_index = 0

    while pixel_count / num_pixels <= 0.5:
        pixel_count += histogram[bin_index]
        bin_index += 1

    return bin_index


def load_frames_from_video_file(
    video_file_path: str,
) -> bool:
    """Split a video file into frames.

    Args:
        `video_file_path`: The local path to the video file.
    Returns:
        A `bool` indicated whether or not we were able to split the video into frames.
    """
    video_file_path = pathlib.Path(video_file_path)

    try:
        video_file = cv.VideoCapture(
            # Oddly enough, `Path.resolve` doesn't return a string.
            str(video_file_path.resolve())
        )
    except Exception as error:
        print(f"Could not open video ('{video_file_path.resolve()}'). ({error})")

        return False

    video_file_images_dir_path = video_file_path.parent.joinpath(
        video_file_path.stem
    )

    try:
        video_file_images_dir_path.mkdir(
            parents=True,
            exist_ok=True,
        )
    except Exception:
        print(f"Could not create image directory ('{video_file_images_dir_path.resolve()}').")

        return False

    frame_count = 1

    # TODO(dsiddy): No infinite loops, please.
    while True:
        try:
            return_value, frame = video_file.read()
        except Exception:
            print('Could not read frame.')

            return False

        frame_file_path = video_file_images_dir_path.joinpath(f'{frame_count}.jpg')

        try:
            cv.imwrite(
                str(frame_file_path),
                frame,
            )
        except Exception as error:
            print(f'Could not write frame. ({error})')

            return False

        frame_count += 1

    return True


def show_histogram(
    frame_file_path: str,
    mask: Union[
        np.ndarray,
        None,
    ] = None,
    num_bins: int = RANGE,
) -> None:
    frame = cv.imread(
        frame_file_path,
    )

    num_pixels = frame.shape[0] * frame.shape[1]

    histogram = cv.calcHist(
        [frame],
        [0],
        mask,
        [num_bins],
        [0, num_bins],
    )

    plt.plot([
        bin_count / num_pixels
        for bin_count in histogram
    ])
    plt.xlabel('brightness')
    plt.xlim([0, num_bins])
    plt.ylabel('proportion of pixels')

    plt.show()


def write_video(
    frames: List[np.ndarray],
    video_file_path: str,
) -> bool:
    writer = cv.VideoWriter(
        video_file_path,
        cv.VideoWriter_fourcc(*'MP4V'),
        24,  # 24,
        (1920, 1080),
    )

    for frame in frames:
        writer.write(frame)

    writer.release()
