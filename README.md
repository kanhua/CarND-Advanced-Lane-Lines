# Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## About this project
This project implements computer vision techniques to identify the lane lines in the images or videos taken by a camera mounted on a car.

### Files
The codes are organized as below:

- [```image_process.py```](./image_process.py): The core algorithms of the project.
- [```project_video_ouput.mp4```](./project_video_output.mp4): The output video file.
- [```image_tests.py```](./image_tests.py): Unit tests.
- [```render_video.py```](./render_video.py): Generate the output video.


## Camera Calibration

I use a series of chessboard images to find the calibration parameters of the camera images. I first prepare an array of "object points" that the corners of the chessboard images should be mapped to. Then, I use ```cv2.findChessBoardCorners()``` to find the coordinates of the chessboard corners. These coordinates are "image points". I then used these image points and object points as the input of ```cv2.calibrteCamera()``` to find the camera matrix and distortion coefficients required to correct the distorted images. After that, the camera matrix and distortion coefficients are used as the input of ```cv2.undistort()``` to perform the image correction. An example of the image correction is shown as below:

![camera calibration](./output_images/camcal.png)

The process of camera calibration and distorted image corrections are wrapped in the class ```image_process.CameraCalibrator```.

Here is an example of correcting the road image using this calibartion:

![road image](./output_images/test1_undistort.png)

## Perspective transformation

I use ```cv2.getPerspectiveTransform()``` and ```cv2.warpPerspective()``` to perform the perspective transformation of road images. Since we are dealing with one camera and car system in this project, the parameters for performing perspective transformation is the same. I simply hardcoded the source and destination coordinates by handpicking these coordinates on a straight road images.
These coordinates are:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 250, 686      | 340, 720        | 
| 583, 458      | 340, 0      |
| 700, 458     | 940, 0      |
| 1060, 686      | 940, 720        |

The resuling images are shown below:

![perspective transformation](./output_images/pers_trans_demo.png)

This process is wrapped in the class ```image_process.PerspectiveTransformer```.


## Lane identification

I made the following binary image masks in order to identify the lanes in an image. The code of this procedure is in ```yellow_white_HLS()```.

#### Yellow filters
This filter selects the pixels in the image with yellow color. The yellow color is defined as the color values between ```[70, 80, 100]``` and ```[105, 255, 255]``` in HSV space. This is inspired by [Yadav's blog post](./https://medium.com/towards-data-science/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3#.9a0h3ccqm).

#### White filters
This filter selects the pixels in the image with white color. The white color is deined as the color values between ```[10, 0, 160]``` and ```[255, 80, 255]```.
This is inspired by [Yadav's blog post](./https://medium.com/towards-data-science/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3#.9a0h3ccqm).

#### S channel of HLS space
This filter selects the pixels in the image that has color values between ```[0, 0, 130]``` and ```[0, 0, 255]``` in HLS color space. This channel works well for selecting yellow and white parts of an image, but it is easily interfered by shadows or bright pavement.

#### L channel of LUV space.
The filter selects the pixels in the image that has color values between ```[0, 0, 130]``` and ```[0, 0, 255]``` in LUV color space and then perform an _NOT_ operation, namely, 

```python
l_binary = channel_select(image, color_space='LUV', thresh=(70, 210), channel=0)
l_binary = np.logical_not(l_binary)
```

This channel selects white and black parts of an image.

#### X gradient
The filter selects the pixels in the image that has high gradient in the x direction in gray scale. In practice, this implemented by applying Sobel matrix to the image using ```cv2.Sobel```. The threshold is selected to be between 0.7 and 1.3.

#### Directional gradient
The filter selects the pixels in the image has image gradient in certain direction, which is defined by 0.7<arctan(grady/gradx)<1.3

The examples of these images are shown in the following graph:

![lane finding](./output_images/binary_mask_breakdown.png)


From the above figure, it can be easily observed that combining L channel, white color, and yellow color can give very reasonable result for this project.

This process in written in the function ```image_process.yellow_white_luv()```.

```python
def yellow_white_luv(image):
	...

    s_binary_2 = hls_select(image, thresh=(130, 255), channel=1)
    white_binary = cf._filter_white(image)
    yellow_binary = cf._filter_yellow(image)

    ksize = 9
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3), sobel_kernel=3)

    l_binary = channel_select(image, color_space='LUV', thresh=(70, 210), channel=0)
    l_binary = np.logical_not(l_binary)

    combined = np.zeros_like(s_binary_2)

    combined[(yellow_binary == 1) | ((l_binary == 1) & (white_binary == 1))] = 1

    ...


```


## Fit the Lanes

I use sliding windows to find the lanes coordinates in the image. This method identifies the lanes using the following procedures:

1. The image is sliced in 9 portions in the y direction.
2. In the top slice, set two windows that are centered at the peaks of histograms.
3. All the color pixels within the windows are set to be the coordinates of the lane for fitting.
4. If there are enough points in the window, recenter the window for the next slice.
5. Iterate step 3 and 4 throughout all the windows.
6. Fit all points found in all the windows with quadratic polynomials.

An example image of this process is shown below:
![fitted_lane](./output_images/demo_bin_fit.jpg)

The codes of these procedures can be found in ```image_process.LandFinder.fit_by_window()```.

After the first set of fitted parameters are found, the lane line points of subsequent frames are located by searching the neighbors of the lane line points of its previous frame. This part of the code can be found in ```image_process.LandFinder._fit_by_prev_fit()```.


## Find the radius of curvature of the lane


### Radius of curvature
The radius of curvature in a cartesian coordinate system is defined as

![eq1](./output_images/radius_curvature.png)

Since our lanes are fitted with a quadratic polynomial:

![eq2](./output_images/second_order_eq.png)

The radius of curvature at _(x0,y0)_ can then be written as:

![eq3](./output_images/radius_curvature_2.png)

I choose the bottom point of the image to caluclate the curvature of radius because it is what the car should immediately repspond to.

### Calibrate the scale of the image

To correct the scale of the camera images, we use the following estimation:

- The width of the lane is 3.7 m.
- The length of the lane is 30 m.

Therefore, each pixel in horizontal direction corresponds to ay=3.7/600=0.0061 m and each pixel in vertical direction corresponds to ax=30/720=0.042 m. The new quadratic polynomial becomes:

![eq4](./output_images/second_order_eq_2.png)

where ```y'=ay*y``` and ```x'=ax*x```.

The procedures of calculating the fitting parameters is in ```image_process.LandFinder._curve_rad_m()```

### Result

Below is a final image with the radius of curvatures and deviations marked on top of the image:

![full_image](./output_images/test1_fp.jpg)

The resulting video is [project_video_output.mp4](./project_video_output.mp4)


## Discussion

Overcoming the shadows and different color of pavements is very challenging in this project. Different colors of lane lines or pavement are likely to fail the lane line identification algorithm.

Another difficulty is getting accurate radius of curvature from the white dashed land lines. The spaces between each segment of white lines causes more error in the fitting. 

To obtain better results of a frame in a video, more different sanity checks could be implemented to further reduce the errors of a single frame.



