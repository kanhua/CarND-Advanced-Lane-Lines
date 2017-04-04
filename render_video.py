import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from moviepy.editor import VideoFileClip

from image_process import *


input_video_file="challenge_video.mp4"

output_video_file="output.mp4"

files = glob.glob("./test_images/test*.jpg")
dummy_img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = get_cv2_img_size(dummy_img)
IMG_FOLDER = './camera_cal/calibration*.jpg'
NX = 9
NY = 5
camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)



lf = LaneFinder()
pip = Pipeline([('cam', camcal), ('undistort', EdgeExtractor()),
                ('pers', PerspectiveTransformer()), ('lane', lf),
                ('inv_pst', PerspectiveTransformer(inv_transform=True))])


pip2 = Pipeline([('clahe',CLAHE()),('cam', camcal), ('undistort', EdgeExtractor()),
                ('pers', PerspectiveTransformer()), ('lane', lf),
                ('inv_pst', PerspectiveTransformer(inv_transform=True))])

def process_image(X):
    nX=np.flip(X,axis=2)
    cv2.imwrite("./image_dump/current_frame.jpg",X)
    nX=pip2.fit_transform(nX)
    return stack_lane_line(X,nX)




white_output = output_video_file
clip1 = VideoFileClip(input_video_file)
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)