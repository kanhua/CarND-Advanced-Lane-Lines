import unittest
from image_process import *
import glob
import cv2

class MyTestCase(unittest.TestCase):

    def setUp(self):

        self.files=glob.glob("./test_images/test*.jpg")
        dummy_img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = get_cv2_img_size(dummy_img)
        IMG_FOLDER = './camera_cal/calibration*.jpg'
        NX = 9
        NY = 5
        self.camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)

    def test_yellow_white_filter(self):

        cf=ColorFilter()
        for idx,f in enumerate(self.files):
            image=cv2.imread(f)
            n_image=cf.transform(image)
            cv2.imwrite("./image_dump/test_yw_%s.jpg"%idx,n_image)

    def run_through_test_images(self):

        lf=LaneFinder()
        pip = Pipeline([('cam', self.camcal), ('undistort', EdgeExtractor()),
                        ('pers', PerspectiveTransformer()), ('lane', lf),
                        ('inv_pst', PerspectiveTransformer(inv_transform=True))])

        for idx,f in enumerate(self.files):
            image=cv2.imread(f)
            transformed_img = pip.fit_transform(image)
            stacked_img = stack_lane_line(image, transformed_img)
            cv2.imwrite("./image_dump/test_fp_%s.jpg" % idx, stacked_img)


    def test_lane_fitting(self):

        pip = Pipeline([('cam', self.camcal), ('undistort', EdgeExtractor()),
                        ('pers', PerspectiveTransformer())])


        for idx,f in enumerate(self.files):
            lane = LaneFinder()
            image=pip.fit_transform(cv2.imread(f))
            savefile="./image_dump/test_lf_%s.jpg" % idx
            lane.visualize(image,savefile=savefile)

    def test_adj_contrast(self):

        clahe=CLAHE()

        nX=clahe.transform(cv2.imread("./test_images/tunnel_1.jpg"))

        cv2.imwrite("./image_dump/adj_tunnel_1.jpg",nX)






if __name__ == '__main__':
    unittest.main()
