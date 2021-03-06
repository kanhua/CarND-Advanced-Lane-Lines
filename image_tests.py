import unittest
from image_process import *
import glob
import cv2
from os.path import join, split, splitext


class MyTestCase(unittest.TestCase):
    def setUp(self):

        self.files = glob.glob("./test_images/test*.jpg")
        dummy_img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = get_cv2_img_size(dummy_img)
        IMG_FOLDER = './camera_cal/calibration*.jpg'
        NX = 9
        NY = 6
        self.camcal = CameraCalibrator(img_size, IMG_FOLDER, NX, NY)

    def test_undistort_sample(self):

        test_image = cv2.imread("./camera_cal/calibration1.jpg")
        undistort_image = self.camcal.transform(test_image)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_image)
        ax[0].set_title("Raw image")
        ax[1].imshow(undistort_image)
        ax[1].set_title("Corrected image")
        fig.savefig("./output_images/camcal.png", bbox_inches='tight')

    def test_undistort_road_sample(self):


        for f in self.files:

            _, fname = split(f)
            fparent, ext = splitext(fname)

            test_image = cv2.imread(f)
            undistort_image = self.camcal.transform(test_image)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.flip(test_image,axis=2))
            ax[0].set_title("Raw image")
            ax[1].imshow(np.flip(undistort_image,axis=2))
            ax[1].set_title("Corrected image")
            fig.savefig("./output_images/{}_undistort.png".format(fparent), bbox_inches='tight')



    def test_binary_sample(self):

        test_image = cv2.imread("./test_images/test1.jpg")
        lane_binary = EdgeExtractor().transform(test_image)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_image)
        ax[0].set_title("Raw image")
        ax[1].imshow(lane_binary)
        ax[1].set_title("Binary mask")
        fig.savefig("./output_images/binary_mask_demo.png", bbox_inches='tight')

    def test_perspective_transform_sample(self):

        test_image = cv2.imread("./test_images/test2.jpg")
        n_image = self.camcal.transform(test_image)
        n_image = PerspectiveTransformer().transform(n_image)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.flip(test_image, axis=2))
        ax[0].set_title("Raw image")
        ax[1].imshow(np.flip(n_image, axis=2))
        ax[1].set_title("Perspective transform")
        fig.savefig("./output_images/pers_trans_demo.png", bbox_inches='tight')

    def test_all_binary_sample(self):

        test_image = cv2.imread("./test_images/test1.jpg")
        plt.imshow(np.flip(test_image, axis=2))
        plt.savefig("./output_images/demo_bin_raw.jpg")
        plt.close()
        n_image = self.camcal.transform(test_image)
        n_image = PerspectiveTransformer().transform(n_image)
        plt.imshow(np.flip(n_image, axis=2))
        plt.savefig("./output_images/demo_bin_trans.jpg")
        plt.close()

        combined, img_comp = yellow_white_luv(n_image)

        for img in img_comp.keys():
            plt.imshow(img_comp[img])
            plt.title(img)
            plt.savefig("./output_images/demo_bin_%s.jpg" % img)
            plt.close()

        plt.imshow(combined)
        plt.savefig("./output_images/demo_bin_all.jpg")

        pip = Pipeline([('cam', self.camcal), ('undistort', EdgeExtractor()),
                        ('pers', PerspectiveTransformer())])

        n_image = pip.fit_transform(test_image)

        lane = LaneFinder()
        savefile = "./output_images/demo_bin_fit.jpg"
        lane.visualize(n_image, savefile=savefile)

    def test_yellow_white_filter(self):

        cf = ColorFilter()
        for idx, f in enumerate(self.files):
            _, fname = split(f)
            fparent, ext = splitext(fname)
            image = cv2.imread(f)
            n_image = cf.transform(image)
            output_fname = join("./image_dump", fparent + "_yw.jpg")
            cv2.imwrite(output_fname, n_image * 255)

    def test_run_through_test_images(self):
        """
        Test the full pipeline of lane detection and labeling
        
        :return: 
        """

        lf = LaneFinder()
        pip = Pipeline([('cam', self.camcal), ('undistort', EdgeExtractor()),
                        ('pers', PerspectiveTransformer()), ('lane', lf),
                        ('inv_pst', PerspectiveTransformer(inv_transform=True))])

        for idx, f in enumerate(self.files):
            _, fname = split(f)
            fparent, ext = splitext(fname)
            image = cv2.imread(f)
            transformed_img = pip.fit_transform(image)
            left_curverad = lf.left_curverad_m
            right_curverad = lf.right_curverad_m
            dev = lf.deviation_m

            stacked_img = stack_lane_line(image, transformed_img, left_curverad, right_curverad, dev)
            output_fname = join("./output_images", fparent + "_fp.jpg")
            cv2.imwrite(output_fname, stacked_img)

    def test_run_through_test_images_with_gaussian(self):
        """
        Test the full pipeline of lane detection and labeling

        :return:
        """

        mlf = MultiPassLaneFinder()
        pip = Pipeline([('cam', self.camcal),
                        ('pers', PerspectiveTransformer()), ('lane', mlf),
                        ('inv_pst', PerspectiveTransformer(inv_transform=True))])

        for idx, f in enumerate(self.files):
            _, fname = split(f)
            fparent, ext = splitext(fname)
            image = cv2.imread(f)
            transformed_img = pip.fit_transform(image)
            left_curverad = mlf.lf.left_curverad_m
            right_curverad = mlf.lf.right_curverad_m
            dev = mlf.lf.deviation_m

            stacked_img = stack_lane_line(image, transformed_img, left_curverad, right_curverad, dev)
            output_fname = join("./output_images", fparent + "_gp.jpg")
            cv2.imwrite(output_fname, stacked_img)

    def test_lane_fitting(self):
        """
        Test the results of LaneFinder class
        
        :return: 
        """

        pip = Pipeline([('cam', self.camcal), ('undistort', EdgeExtractor()),
                        ('pers', PerspectiveTransformer())])

        for idx, f in enumerate(self.files):
            _, fname = split(f)
            fparent, ext = splitext(fname)

            lane = LaneFinder()
            image = pip.fit_transform(cv2.imread(f))
            savefile = join("./image_dump", fparent + "_lf.jpg")

            lane.visualize(image, savefile=savefile)

    def test_adj_contrast(self):
        """
        Test adjusting the contract with CLAHE
        
        :return: 
        """

        clahe = CLAHE()

        nX = clahe.transform(cv2.imread("./test_images/tunnel_1.jpg"))

        cv2.imwrite("./image_dump/adj_tunnel_1.jpg", nX)


if __name__ == '__main__':
    unittest.main()
