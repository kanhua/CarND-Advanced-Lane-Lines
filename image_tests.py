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
        NY = 6
        self.camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)

    def undistort_sample(self):

        test_image=cv2.imread("./camera_cal/calibration1.jpg")
        undistort_image=self.camcal.transform(test_image)

        fig,ax=plt.subplots(1,2)
        ax[0].imshow(test_image)
        ax[0].set_title("Raw image")
        ax[1].imshow(undistort_image)
        ax[1].set_title("Corrected image")
        fig.savefig("./output_images/camcal.png")

    def binary_sample(self):

        test_image=cv2.imread("./test_images/test1.jpg")
        lane_binary=EdgeExtractor().transform(test_image)

        fig,ax=plt.subplots(1,2)
        ax[0].imshow(test_image)
        ax[0].set_title("Raw image")
        ax[1].imshow(lane_binary)
        ax[1].set_title("Binary mask")
        fig.savefig("./output_images/binary_mask_demo.png")

    def perspective_transform_sample(self):

        test_image=cv2.imread("./test_images/straight_lines1.jpg")
        n_image=self.camcal.transform(test_image)
        n_image=PerspectiveTransformer().transform(n_image)

        fig,ax=plt.subplots(1,2)
        ax[0].imshow(np.flip(test_image,axis=2))
        ax[0].set_title("Raw image")
        ax[1].imshow(np.flip(n_image,axis=2))
        ax[1].set_title("Perspective transform")
        fig.savefig("./output_images/pers_trans_demo.png")


    def test_yellow_white_filter(self):

        cf=ColorFilter()
        for idx,f in enumerate(self.files):
            image=cv2.imread(f)
            n_image=cf.transform(image)
            cv2.imwrite("./image_dump/test_yw_%s.jpg"%idx,n_image*255)

    def run_through_test_images(self):

        lf=LaneFinder()
        pip = Pipeline([('cam', self.camcal), ('undistort', EdgeExtractor()),
                        ('pers', PerspectiveTransformer()), ('lane', lf),
                        ('inv_pst', PerspectiveTransformer(inv_transform=True))])

        for idx,f in enumerate(self.files):
            image=cv2.imread(f)
            transformed_img = pip.fit_transform(image)
            left_curverad=lf.left_curverad_m
            right_curverad=lf.right_curverad_m

            stacked_img = stack_lane_line(image, transformed_img,left_curverad,right_curverad)
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
