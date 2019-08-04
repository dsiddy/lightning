# `lightning`

On July 23, 2019, [the most spectacular lightning storm that I've ever seen](https://www.spokesman.com/stories/2019/jul/23/more-than-20-fires-erupt-around-spokane-county-tue/) rolled through Spokane. I managed to capture 15 minutes' worth of video. Rather than pick out the most striking stills manually, I took the opportunity to acquaint myself with OpenCV, and wrote some code that automatically extracts frames whose brightness exceeds a given threshold. (Well, more precisely, it extracts a frame if the proportion of the frame's pixels that fall within prescribed regional and brightness bounds exceeds a given threshold.)

For the 10-second, potentially seizure-inducing result of this endeavor, check out [`data/flashes.mp4`](data/flashes.mp4). :zap:

## requirements

* [`matplotlib`](https://matplotlib.org/) (but only for the `show_histogram` function)
* [`opencv-python`](https://github.com/skvark/opencv-python)

## resources

* [OpenCV: Histograms - 1 : Find, Plot, Analyze !!!
](https://docs.opencv.org/master/d1/db7/tutorial_py_histogram_begins.html): The tutorial that brought me up to speed.
* [Reading and Writing Images and Video](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html)
* [`cv2.cvtColor`](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor): Documentation for the method that I used to convert color images to grayscale ones.
