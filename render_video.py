import numpy as np
import cv2
import glob
from sklearn.pipeline import Pipeline

from moviepy.editor import VideoFileClip

import matplotlib.pyplot as plt

from image_process import LaneFinder, EdgeExtractor, \
    PerspectiveTransformer, CLAHE, stack_lane_line, get_cv2_img_size, CameraCalibrator, MultiPassLaneFinder

input_video_file = "harder_challenge_video.mp4"

output_video_file = "harder_challenge_video_output.mp4"

files = glob.glob("./test_images/test*.jpg")
dummy_img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = get_cv2_img_size(dummy_img)
IMG_FOLDER = './camera_cal/calibration*.jpg'
NX = 9
NY = 6
camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)

lf = LaneFinder()
pip = Pipeline([('cam', camcal), ('undistort', EdgeExtractor()),
                ('pers', PerspectiveTransformer()), ('lane', lf),
                ('inv_pst', PerspectiveTransformer(inv_transform=True))])

pip2 = Pipeline([('clahe', CLAHE()), ('cam', camcal), ('undistort', EdgeExtractor()),
                 ('pers', PerspectiveTransformer()), ('lane', lf),
                 ('inv_pst', PerspectiveTransformer(inv_transform=True))])

mlf = MultiPassLaneFinder()
pip3 = Pipeline([('cam', camcal),
                 ('pers', PerspectiveTransformer()), ('lane', mlf),
                 ('inv_pst', PerspectiveTransformer(inv_transform=True))])


def process_image(X):
    nX = np.flip(X, axis=2)
    # cv2.imwrite("./image_dump/current_frame.jpg",X)
    try:
        nX = pip3.fit_transform(nX)
    except:
        plt.imshow(X)
        plt.savefig("problem_image.png")

    return stack_lane_line(X, nX, mlf.lf.left_curverad_m, mlf.lf.right_curverad_m, mlf.lf.deviation_m)


def image_break_down(X):
    nX = np.flip(X, axis=2)
    fitted_img = np.zeros_like(nX)
    mask_img = np.zeros_like(nX)
    curve_fit_img = np.zeros_like(nX)
    final_img = nX

    try:
        fitted_img = pip3.fit_transform(nX)
        mask_img = np.dstack((mlf.fitted_img, mlf.fitted_img, mlf.fitted_img))
        curve_fit_img = mlf.lf.visualize(mlf.fitted_img)
    except:
        plt.imshow(X)
        plt.savefig("problem_image.png")

    final_img = stack_lane_line(X, fitted_img, mlf.lf.left_curverad_m, mlf.lf.right_curverad_m, mlf.lf.deviation_m)
    left_image = np.concatenate((final_img, curve_fit_img), axis=0)
    right_image = np.concatenate((mask_img, fitted_img), axis=0)
    all_image = np.concatenate((left_image, right_image), axis=1)

    return all_image


white_output = output_video_file
clip1 = VideoFileClip(input_video_file)

white_clip = clip1.fl_image(image_break_down)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
