import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from moviepy.editor import VideoFileClip


class CameraCalibrator(BaseEstimator):
    def __init__(self, img_size, img_folder, nx, ny, obj_img_cache="obj_img_cache.p", recalc=False):
        self.img_size = img_size

        if os.path.exists(obj_img_cache) == True and recalc == False:
            cache = pickle.load(open(obj_img_cache, 'rb'))
            self.objpoints = cache['objpoints']
            self.imgpoints = cache['imgpoints']

        else:
            self.objpoints, self.imgpoints = get_calibration_factors(image_folder=img_folder,
                                                                     nx=nx, ny=ny)
            with open(obj_img_cache, 'wb') as fp:
                pickle.dump({'objpoints': self.objpoints, 'imgpoints': self.imgpoints}, fp)

        self.retval, self.cameraMatrix, \
        self.distCoeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints,
                                                                      self.imgpoints, self.img_size, None, None)

    def transform(self, img):
        assert get_cv2_img_size(img) == self.img_size
        undist_img = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)

        return undist_img

    def fit(self, X, y):

        return self


class PerspectiveTransformer(BaseEstimator):
    def __init__(self, src=None, dst=None, inv_transform=False):

        default_src = [
            (250, 686),  # left, bottom
            (583, 458),  # left, top
            (700, 458),  # right, top
            (1060, 686)]  # right, bottom

        default_dst = [
            [300, 720],
            [300, 0],
            [800, 0],
            [800, 720]
        ]

        if src is None:
            self.src = np.array(default_src).astype(np.float32)
        else:
            self.src = src

        if dst is None:
            self.dst = np.array(default_dst).astype(np.float32)
        else:
            self.dst = dst

        self.inv_transform = inv_transform

    def transform(self, img):
        if self.inv_transform:
            return warper(img, self.dst, self.src)
        else:
            return warper(img, self.src, self.dst)

    def fit(self, X, y):
        return self

    def get_M(self):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        return M

    def get_invM(self):
        invM = cv2.getPerspectiveTransform(self.dst, self.src)
        return invM


def process_image(image):
    dummy_img = cv2.imread('./camera_cal/calibration1.jpg')
    img_size = get_cv2_img_size(dummy_img)
    IMG_FOLDER = './camera_cal/calibration*.jpg'
    NX = 9
    NY = 5
    camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)
    pip = Pipeline([('cam', camcal), ('undistort', EdgeExtractor()),
                    ('pers', PerspectiveTransformer()), ('lane', LaneFinder()),
                    ('inv_pst', PerspectiveTransformer(inv_transform=True))])
    transformed_img = pip.fit_transform(image)

    stacked_img = stack_lane_line(image, transformed_img)

    return stacked_img


def get_cv2_img_size(img):
    assert isinstance(img, np.ndarray)

    return img[:, :, 0].T.shape[:2]


def stack_lane_line(road_img, lane_img):
    # Combine the result with the original image
    result = cv2.addWeighted(road_img, 1, lane_img, 0.3, 0)
    return result


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def abs_sobel_thresh(img, orient='x', sobel_kernel=1, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2),
                  to_gray=True,sx_thresh=(30,100),sy_thesh=(30,100)):
    # Grayscale
    if to_gray:
        img = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, thresh=(0, 255),channel=2):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def get_calibration_factors(image_folder='./camera_cal/calibration*.jpg', nx=9, ny=5):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(image_folder)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)

    return objpoints, imgpoints


def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img[:, :, 0].T.shape[:2],
                                                                         None, None)
    undist = cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)
    # undist = np.copy(img)  # Delete this line
    return undist


class EdgeExtractor(BaseEstimator):
    def __init__(self, s_thresh=(100, 255), sx_thresh=(50, 200)):
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def transform(self, X):
        #_, n_img = edge_pipeline(X, self.s_thresh, self.sx_thresh)

        n_img=edge_pipline_v2(X)

        return n_img

    def fit(self, X, y):
        return self

    def visualize(self, img):
        color_img, _ = edge_pipeline(img, self.s_thresh, self.sx_thresh)

        return color_img


def edge_pipeline(img, s_thresh=(170, 255), sx_thresh=(50, 200)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    dir_binary = dir_threshold(l_channel, thresh=(0.7, 1.3), to_gray=False)

    single_channel_binary = np.zeros_like(scaled_sobel)
    single_channel_binary[((s_binary == 1) & (dir_binary == 1)) | (sxbinary == 1)] = 1

    return color_binary, single_channel_binary


def edge_pipline_v2(difficult_image):

    hls = cv2.cvtColor(difficult_image, cv2.COLOR_RGB2HLS)

    s_binary_1 = hls_select(difficult_image, thresh=(100, 200), channel=1)

    s_binary_2 = hls_select(difficult_image, thresh=(130, 255), channel=2)

    ksize = 3
    gradx = abs_sobel_thresh(difficult_image, orient='x', sobel_kernel=ksize, thresh=(20, 100))


    grady = abs_sobel_thresh(difficult_image, orient='y', sobel_kernel=ksize, thresh=(20, 100))


    dir_binary = dir_threshold(difficult_image, thresh=(0.9, 1.2), sobel_kernel=3)


    mag_binary = mag_thresh(difficult_image, mag_thresh=(30, 100))


    combined = np.zeros_like(dir_binary)
    combined[((dir_binary == 1) & (mag_binary == 1)) | \
             ((gradx == 1) & (grady == 1)) | ((s_binary_2 == 1))] = 1

    return combined


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


class LaneFinder(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.leftx = None
        self.lefty = None
        pass

    def fit(self, X, y=None):

        if self.leftx is None:
            self.leftx, self.lefty, self.left_fit, self.rightx, \
            self.righty, self.right_fit, self.out_img, self.nonzerox, \
            self.nonzeroy, self.left_lane_inds, self.right_lane_inds = find_lane_points(X)

            self.ploty = np.linspace(0, X.shape[0] - 1, X.shape[0])
            self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
            self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

        else:
            self._fit_next(X)

        self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        return self

    def _fit_next(self, X, y=None):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        self.nonzero = X.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        margin = 100
        self.left_lane_inds = (
            (self.nonzerox > (
            self.left_fit[0] * (self.nonzeroy ** 2) + self.left_fit[1] * self.nonzeroy + self.left_fit[2] - margin)) & (
                self.nonzerox < (
                self.left_fit[0] * (self.nonzeroy ** 2) + self.left_fit[1] * self.nonzeroy + self.left_fit[
                    2] + margin)))
        self.right_lane_inds = (
            (self.nonzerox > (
            self.right_fit[0] * (self.nonzeroy ** 2) + self.right_fit[1] * self.nonzeroy + self.right_fit[
                2] - margin)) & (
                self.nonzerox < (
                self.right_fit[0] * (self.nonzeroy ** 2) + self.right_fit[1] * self.nonzeroy + self.right_fit[
                    2] + margin)))

        # Again, extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, X.shape[0] - 1, X.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

    def transform(self, X, y=None):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(X).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        return color_warp

    def visualize(self, X, y=None):

        self.fit(X)
        ploty = np.linspace(0, X.shape[0] - 1, X.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        plt.imshow(self.out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)


def render_video(input_video_file, output_video_file):
    white_output = output_video_file
    clip1 = VideoFileClip(input_video_file)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


def find_lane_points(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) \
                          & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) \
                           & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return leftx, lefty, left_fit, rightx, righty, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds


if __name__ == "__main__":
    test_image = "./test_images/straight_lines2.jpg"

    n_image = process_image(cv2.imread(test_image))

    cv2.imwrite("test_out.png", n_image)
