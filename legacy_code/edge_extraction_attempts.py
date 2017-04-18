"""
This script collects legacy edge extraction attempts

"""

from image_process import *

def yellow_white_hls(image):
    cf = ColorFilter()
    wy_binary = cf.transform(image)

    s_binary_2 = hls_select(image, thresh=(130, 255), channel=2)
    white_binary = cf._filter_white(image)
    yellow_binary = cf._filter_yellow(image)

    ksize = 9
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3), sobel_kernel=3)

    combined = np.zeros_like(s_binary_2)
    combined[((s_binary_2 == 1) & (white_binary == 1)) | (yellow_binary == 1) | (
        (gradx == 1) & (dir_binary == 1) & (white_binary == 1))] = 1

    img_comp = {}
    img_comp['s_binary'] = s_binary_2
    img_comp['white'] = white_binary
    img_comp['yellow'] = yellow_binary
    img_comp['gradx'] = gradx
    img_comp['dir'] = dir_binary

    return combined, img_comp


def yellow_white_hls_2(image):
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
    combined[((s_binary_2 == 1) & (gradx == 1)) | (yellow_binary == 1) | (
        (gradx == 1) & (dir_binary == 1) & (white_binary == 1))] = 1

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


def yellow_white_hls_3(image):
    cf = ColorFilter()
    wy_binary = cf.transform(image)

    s_binary_2 = hls_select(image, thresh=(130, 255), channel=2)
    white_binary = cf._filter_white(image)
    yellow_binary = cf._filter_yellow(image)

    ksize = 9
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3), sobel_kernel=3)

    combined = np.zeros_like(s_binary_2)
    combined[((s_binary_2 == 1) & (white_binary == 1)) | (yellow_binary == 1) | (
        (gradx == 1) & (dir_binary == 1) & (white_binary == 1))] = 1

    img_comp = {}
    img_comp['s_binary'] = s_binary_2
    img_comp['white'] = white_binary
    img_comp['yellow'] = yellow_binary
    img_comp['gradx'] = gradx
    img_comp['dir'] = dir_binary

    return combined, img_comp


def edge_pipeline(img, s_thresh=(170, 255), sx_thresh=(50, 200)):
    """
    The fist version of edge extraction pipeline. This is adapted from
    Udacity course material

    :param img: 
    :param s_thresh: 
    :param sx_thresh: 
    :return: 
    """

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


def edge_pipeline_v3(difficult_image):
    cf = ColorFilter()
    wy_binary = cf.transform(difficult_image)

    ksize = 3
    gradx = abs_sobel_thresh(difficult_image, orient='x', sobel_kernel=ksize, thresh=(20, 100))

    grady = abs_sobel_thresh(difficult_image, orient='y', sobel_kernel=ksize, thresh=(20, 100))

    dir_binary = dir_threshold(difficult_image, thresh=(0.9, 1.2), sobel_kernel=3)

    mag_binary = mag_thresh(difficult_image, mag_thresh=(30, 100))

    combined = np.zeros_like(dir_binary)
    combined[((dir_binary == 1) & (mag_binary == 1)) | (wy_binary == 1)] = 1

    return combined