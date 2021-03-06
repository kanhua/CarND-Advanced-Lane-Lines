import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
from copy import copy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class CameraCalibrator(BaseEstimator, TransformerMixin):
    """
    This class runs the camera images calibration and transform new images using the calibration parameters
    
    """

    def __init__(self, img_size=None, img_folder=None, nx=None, ny=None, obj_img_cache="obj_img_cache.p", recalc=False):
        self.img_size = img_size

        if os.path.exists(obj_img_cache) == True and recalc == False:
            with open(obj_img_cache,'rb') as fp:
                cache = pickle.load(fp)
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

    def transform(self, img, y=None):
        assert get_cv2_img_size(img) == self.img_size
        undist_img = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)

        return undist_img

    def fit(self, X, y=None):

        return self


class CLAHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None, **fit_params):
        hsv_X = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)

        clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(4, 4))

        hsv_X[:, :, 2] = clahe.apply(hsv_X[:, :, 2])

        nX = cv2.cvtColor(hsv_X, cv2.COLOR_HSV2RGB)

        return nX

    def fit(self, X, y=None, **fit_params):
        return self


class PerspectiveTransformer(BaseEstimator, TransformerMixin):
    """
    This class performs perspective transformation in both (foward and reverse) direction
    
    """

    def __init__(self, dst=[
        [340, 720],
        [340, 0],
        [940, 0],
        [940, 720]
    ],
                 src=[(250, 686),  # left, bottom
                      (583, 458),  # left, top
                      (700, 458),  # right, top
                      (1060, 686)], inv_transform=False):

        self.src = np.array(src).astype(np.float32)

        self.dst = np.array(dst).astype(np.float32)

        self.inv_transform = inv_transform

    def transform(self, img, y=None):
        if self.inv_transform:
            return warper(img, self.dst, self.src)
        else:
            return warper(img, self.src, self.dst)

    def fit(self, X, y=None):
        return self

    def get_M(self):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        return M

    def get_invM(self):
        invM = cv2.getPerspectiveTransform(self.dst, self.src)
        return invM


class ColorFilter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _color_mask(self, X, low, high):
        nX = np.copy(X)
        mask = cv2.inRange(nX, low, high)
        nmask = np.zeros_like(mask)
        nmask[mask == 255] = 1
        return nmask

    def _filter_white(self, X):
        hsv = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
        white_hsv_low = np.array([10, 0, 160])
        white_hsv_high = np.array([255, 80, 255])
        white_filtered_image = self._color_mask(hsv, white_hsv_low, white_hsv_high)
        return white_filtered_image

    def _filter_yellow(self, X):
        hsv = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
        yellow_hsv_low = np.array([70, 80, 100])
        yellow_hsv_high = np.array([105, 255, 255])
        yellow_filtered_image = self._color_mask(hsv, yellow_hsv_low, yellow_hsv_high)
        return yellow_filtered_image

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, mask=True):
        wh = self._filter_white(X)
        yl = self._filter_yellow(X)
        yellow_white_binary = np.zeros_like(wh)
        yellow_white_binary[(wh == 1) | (yl == 1)] = 1
        if mask:
            return yellow_white_binary
        res = cv2.bitwise_and(X, X, mask=yellow_white_binary)
        return res


def get_cv2_img_size(img):
    assert isinstance(img, np.ndarray)

    return img[:, :, 0].T.shape[:2]


def stack_lane_line(road_img, lane_img, left_curverad=None, right_curverad=None, deviation=None):
    # Combine the result with the original image

    font = cv2.FONT_HERSHEY_SIMPLEX
    if left_curverad is not None:
        n_road_img = cv2.putText(road_img,
                                 "left r: {:.2f} m, right r: {:.2f} m, dev: {:.2f} m".format(left_curverad,
                                                                                             right_curverad, deviation),
                                 (20, 40), font, 1, (255, 255, 255), 2,
                                 cv2.LINE_AA)
    else:
        n_road_img = np.copy(road_img)
    result = cv2.addWeighted(n_road_img, 1, lane_img, 0.3, 0)
    return result


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), to_gray=True):
    if to_gray == True:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
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


def abs_sobel_thresh(img, orient='x',
                     sobel_kernel=1, thresh=(0, 255), to_gray=True):
    # Convert to grayscale
    if to_gray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel = None
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
                  to_gray=True, sx_thresh=(30, 100), sy_thesh=(30, 100)):
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


def hls_select(img, thresh=(0, 255), channel=2):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def hsv_select(img, thresh=(0, 255), channel=2):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hls[:, :, channel]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def channel_select(img, color_space='HSV', thresh=(0, 255), channel=2):
    """
    A more general version of generating a mask from a single channel
    
    :param img: image array 
    :param color_space: color space code, e.g., 'HSV','Lab','HLS'
    :param thresh: tuple of (min_bound, max_bound)
    :param channel: integer. Color channel to selelct
    :return: binary mask array
    """
    color_code = eval('cv2.COLOR_RGB2{}'.format(color_space))
    new_img = cv2.cvtColor(img, color_code)
    s_channel = new_img[:, :, channel]
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


class EdgeExtractor(BaseEstimator):
    def __init__(self, s_thresh=(100, 255), sx_thresh=(50, 200)):
        self.s_thresh = s_thresh
        self.sx_thresh = sx_thresh

    def transform(self, X):
        # _, n_img = edge_pipeline(X, self.s_thresh, self.sx_thresh)

        #n_img, _ = yellow_white_hls(X)

        n_img1,_=yellow_white_luv(X)
        #n_img,_=yellow_white_luv_gradx(X)
        n_img2,_=simple_luv_lab(X)


        #n_img=n_img1|n_img2
        return n_img1

    def fit(self, X, y):
        return self

    def visualize(self, img):

        color_img, _ = yellow_white_luv(img)

        return color_img



def yellow_white_luv(image):

    cf = ColorFilter()
    wy_binary = cf.transform(image)

    white_binary = cf._filter_white(image)
    yellow_binary = cf._filter_yellow(image)


    l_binary = channel_select(image, color_space='LUV', thresh=(70, 210), channel=0)
    l_binary = np.logical_not(l_binary)

    combined = np.zeros_like(l_binary)

    combined[(yellow_binary == 1) | ((l_binary == 1) & (white_binary == 1))] = 1

    # Perform l_binary+white_channel
    l_w = np.zeros_like(l_binary)
    l_w[(l_binary == 1) & (white_binary == 1)] = 1

    img_comp = {}
    img_comp['white'] = white_binary
    img_comp['yellow'] = yellow_binary
    img_comp['l_binary'] = l_binary
    img_comp['l_bin_w'] = l_w

    return combined.astype(np.uint8), img_comp

def yellow_white_luv_gradx(image):
    cf = ColorFilter()
    wy_binary = cf.transform(image)

    s_binary_2 = hls_select(image, thresh=(130, 255), channel=1)
    white_binary = cf._filter_white(image)
    yellow_binary = cf._filter_yellow(image)

    ksize = 9
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3), sobel_kernel=3)

    l_binary = channel_select(image, color_space='LUV', thresh=(70, 210), channel=0)
    l_binary = np.logical_not(l_binary)

    combined = np.zeros_like(s_binary_2)

    combined[(yellow_binary == 1) | ((l_binary == 1) & (white_binary == 1))|(gradx==1)] = 1

    # Perform l_binary+white_channel
    l_w = np.zeros_like(s_binary_2)
    l_w[(l_binary == 1) & (white_binary == 1)] = 1

    img_comp = {}
    img_comp['s_binary'] = s_binary_2
    img_comp['white'] = white_binary
    img_comp['yellow'] = yellow_binary
    img_comp['gradx'] = gradx
    img_comp['dir'] = dir_binary
    img_comp['l_binary'] = l_binary
    img_comp['l_bin_w'] = l_w

    return combined, img_comp

def simple_luv_lab(image):

    b_binary = channel_select(image, color_space='Lab', thresh=(0, 120), channel=2)
    l_binary = channel_select(image, color_space='LUV', thresh=(170, 255), channel=0)

    img_comp={}
    img_comp['b_binary']=b_binary
    img_comp['l_binary']=l_binary
    combined=b_binary|l_binary

    return combined.astype(np.uint8),img_comp



def load_straight_lane_image():

    dummy_img = cv2.imread('./camera_cal/calibration1.jpg')
    img_size = get_cv2_img_size(dummy_img)
    IMG_FOLDER = './camera_cal/calibration*.jpg'
    NX = 9
    NY = 6
    camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)

    all_pipes = [('cam', camcal), ('undistort', EdgeExtractor()),
                 ('pers', PerspectiveTransformer())]
    pip = Pipeline(all_pipes)
    lane_cal_image = pip.transform(cv2.imread("./test_images/straight_lines1.jpg"))

    return lane_cal_image


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


class LaneFinder(BaseEstimator, TransformerMixin):
    def __init__(self):

        self.leftx_base = None
        self.rightx_base = None

        self.raw_image = None

        self.lane_coords = {}
        self.fitted_param = {}

        self.left_curverad = None
        self.righ_curverad = None



    def fit(self, X, y=None):

        assert X.ndim <= 2

        self.raw_image = np.copy(X)

        self._reset_out_img(X)

        if not self.fitted_param:
            self._find_base(X)
            lane_coords = self.fit_by_window(X)

            fitted_param = self._curve_fit(X, **lane_coords)

        else:
            # if self.fitted_param exists, look for lane points by searching the proximity of the previously fitted lane

            lane_coords = self._fit_by_prev_fit(X, self.fitted_param['left_fit'], self.fitted_param['right_fit'])

            fitted_param = self._curve_fit(X, **lane_coords)

            # Sanity check 1
            #if self._base_shifted(fitted_param['left_fitx'], fitted_param['right_fitx']):
            #    self._find_base(X)
            #    lane_coords = self.fit_by_window(X)
            #    fitted_param = self._curve_fit(X, **lane_coords)

        #Check fitted lane separation
        if not self._check_lane_separation(fitted_param['left_fitx'],fitted_param['right_fitx']):
            if self.fitted_param:
                fitted_param=self.fitted_param
                lane_coords=self.lane_coords


        # Calculate the curvature

        left_curverad = self._curve_rad(np.max(fitted_param['ploty']), fitted_param['left_fit'])
        right_curverad = self._curve_rad(np.max(fitted_param['ploty']), fitted_param['right_fit'])

        # if self._lr_curve_rad_cmp(left_curverad, right_curverad,margin_ratio=0.5):
        #     if self.left_curverad is not None:
        #         left_curverad = self.left_curverad
        #         right_curverad = self.righ_curverad
        #         fitted_param = self.fitted_param

        left_curverad_m, right_curverad_m, deviation = self._curve_rad_m(X, np.max(fitted_param['ploty']),
                                                                         **lane_coords)

        self.lane_coords = copy(lane_coords)

        self.left_curverad = left_curverad
        self.righ_curverad = right_curverad

        self.left_curverad_m = left_curverad_m
        self.right_curverad_m = right_curverad_m

        self.deviation_m = deviation

        self.fitted_param = copy(fitted_param)

        return self

    def _base_shifted(self, leftx, rightx, margin=50):
        """
        Check if the found lane shifts too much
        
        :param leftx: 
        :param rightx: 
        :param margin: 
        :return: 
        """

        if np.abs(leftx[-1] - self.leftx_base) > margin or np.abs(rightx[-1] - self.rightx_base) > margin:
            return True
        else:
            return False

    def _check_lane_separation(self,leftx,rightx,ratio_thres=0.5):
        """
        Check if the fitted lane separation is correct. Return False if anything wrong happens
        
        :param leftx: 
        :param rightx: 
        :param ratio_thres: 
        :return: 
        """

        sep_d=rightx-leftx

        if np.any(sep_d<=0):
            return False

        #print("warning: _check_lane_separation being hacked for debugging!")
        if (np.max(sep_d)-np.min(sep_d))> (np.mean(sep_d)*ratio_thres):
            return False

        return True



    def _lr_curve_rad_cmp(self, prev_curve_rad, curr_curve_rad, margin_ratio=0.1):
        if np.abs(prev_curve_rad / curr_curve_rad - 1) > margin_ratio:
            return True
        else:
            return False

    def _reset_out_img(self, X):
        self.out_img = np.dstack((X, X, X))
        return self.out_img

    def _find_base(self, calibration_X):

        # Assuming you have created a warped binary image called "X"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(calibration_X[int(calibration_X.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    def fit_by_window(self, X):
        # Assuming you have created a warped binary image called "X"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(X[int(X.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and  visualize the result

        self._reset_out_img(X)

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(X.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = X.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
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
            win_y_low = X.shape[0] - (window + 1) * window_height
            win_y_high = X.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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

        lane_coords = {}
        lane_coords['leftx'] = leftx
        lane_coords['lefty'] = lefty
        lane_coords['rightx'] = rightx
        lane_coords['righty'] = righty

        return lane_coords

    def _curve_fit(self, X, leftx, lefty, rightx, righty):

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, X.shape[0] - 1, X.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # self.left_curverad=self._curve_rad(np.max(self.ploty),self.left_fit)
        # self.right_curverad=self._curve_rad(np.max(self.ploty),self.right_fit)
        fitted_param = {}
        fitted_param['left_fit'] = left_fit
        fitted_param['right_fit'] = right_fit
        fitted_param['left_fitx'] = left_fitx
        fitted_param['right_fitx'] = right_fitx
        fitted_param['ploty'] = ploty

        return fitted_param

    def _curve_rad(self, y_eval, poly_fit_param):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        curverad = ((1 + (2 * poly_fit_param[0] * y_eval + \
                          poly_fit_param[1]) ** 2) ** 1.5) / np.absolute(2 * poly_fit_param[0])

        return curverad

    def _curve_rad_m(self, X, y_eval, leftx, lefty, rightx, righty):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

        ploty = np.linspace(0, X.shape[0] - 1, X.shape[0])
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        # Get vehicle deviation
        mid_point = X.shape[1] / 2.0 * xm_per_pix

        bottom_y = ploty[-1] * ym_per_pix
        left_fitx = left_fit_cr[0] * bottom_y ** 2 + left_fit_cr[1] * bottom_y + left_fit_cr[2]
        right_fitx = right_fit_cr[0] * bottom_y ** 2 + right_fit_cr[1] * bottom_y + right_fit_cr[2]

        car_point = (right_fitx + left_fitx) / 2.0

        deviation = car_point - mid_point

        return left_curverad, right_curverad, deviation

    def _fit_by_prev_fit(self, X, left_fit, right_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = X.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (
                left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (
                    left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[
                        2] + margin)))
        right_lane_inds = (
            (nonzerox > (
                right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[
                    2] - margin)) & (
                nonzerox < (
                    right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[
                        2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        lane_coords = {}
        lane_coords['leftx'] = leftx
        lane_coords['lefty'] = lefty
        lane_coords['rightx'] = rightx
        lane_coords['righty'] = righty

        return lane_coords

    def gaussian_filter(self, left_on, right_on):
        """
        Generate the gaussian distribution centered on the lane line based on fitted results.

        :param left_on: True if show the left lane line
        :param right_on: True if show the right lane line
        :return: the filtered image
        """
        new_image = np.empty_like(self.raw_image).astype(np.float32)
        leftx = self.fitted_param['left_fitx']
        rightx = self.fitted_param['right_fitx']

        sigma = 50
        # coef=1/np.sqrt(2*np.pi*sigma**2)
        coef = 1

        for x in range(new_image.shape[1]):
            new_image[:, x] = coef*left_on * np.exp(-(x - leftx) ** 2 / (2 * sigma ** 2)) + coef*right_on * np.exp(
                -(x - rightx) ** 2 / (2 * sigma ** 2))

        return new_image


    def transform(self, X, y=None):

        assert X.ndim<=2

        if self.fitted_param is None:
            raise ValueError("Not fitted yet.")


        left_fitx = self.fitted_param['left_fitx']
        right_fitx = self.fitted_param['right_fitx']
        ploty = self.fitted_param['ploty']

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(X).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        return color_warp

    def visualize(self, X, y=None, savefile="cache.png"):

        self.fit(X)
        left_fitx = self.fitted_param['left_fitx']
        right_fitx = self.fitted_param['right_fitx']
        ploty = self.fitted_param['ploty']

        self.out_img[self.lane_coords['lefty'], self.lane_coords['leftx']] = [255, 0, 0]
        self.out_img[self.lane_coords['righty'], self.lane_coords['rightx']] = [0, 0, 255]
        plt.imshow(self.raw_image, cmap='gray')
        plt.imshow(self.out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig(savefile)
        plt.close()
        return self.out_img


class MultiPassLaneFinder(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.lf=LaneFinder()


    def fit(self, X, y=None):

        yw_image,_=yellow_white_luv(X)

        self.lf.fit(yw_image)
        left_g_img=self.lf.gaussian_filter(left_on=True,right_on=False)


        left_threshold=0.5
        lab_img = cv2.cvtColor(X, cv2.COLOR_RGB2LAB)
        b_img_sobel = abs_sobel_thresh(lab_img[:, :, 2], orient='x',
                                 sobel_kernel=5, thresh=(10, 100), to_gray=False)

        b_img=b_img_sobel*left_g_img

        luv_img = cv2.cvtColor(X, cv2.COLOR_RGB2LUV)
        sobel_img = abs_sobel_thresh(luv_img[:, :, 0], orient='x', sobel_kernel=5, thresh=(10, 80), to_gray=False)

        right_g_img=self.lf.gaussian_filter(left_on=False,right_on=True)
        right_threshold=0.7

        l_img=sobel_img*right_g_img


        fitted_img=l_img+b_img

        fitted_img[fitted_img>=right_threshold]=1
        fitted_img[fitted_img<right_threshold]=0

        fitted_img=fitted_img.astype(np.uint8)

        plt.imshow(fitted_img)
        plt.savefig("fitted_img.png")
        plt.close()

        #self.lf.fit(fitted_img)
        nlf=LaneFinder()

        self.lf.visualize(fitted_img)

        self.fitted_img=fitted_img

        return self

    def transform(self,X,y=None):

        self.fit(X)

        return self.lf.transform(self.fitted_img)


    def fit_transform(self, X, y=None, **fit_params):


        return self.transform(X)










if __name__ == "__main__":
    pass
