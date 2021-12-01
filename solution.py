import os
import cv2
import sys
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import phasepack.phasecong as pc
from skimage.metrics import structural_similarity

class bcolors():
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class weld_removal():
    def __init__(self, path_fume, path_ground_truth, colors):
        self.path_fume = path_fume
        self.path_ground_truth = path_ground_truth
        self.color = colors
        self.img1 = None
        self.img2 = None
        self.input1 = None
        self.input2 = None
        self.input1normalised = None
        self.input2normalised = None
        self.input_images = None
        self.pyramid_height = 10
        self.gaussians = None
        self.fused = None
        self.weights = None
        self.fused_image = None
        self.psnr_list = []
        self.ssim_list = []

    def white_balance(self, image=None):
        if image is None: return None
        image = np.float64(image)
        B,G,R = (image[:, :, 0],image[:, :, 1],image[:, :, 2])
        meanR,meanG,meanB = (np.mean(R),np.mean(G),np.mean(B))
        meanRGB = np.array([meanR, meanG, meanB])
        grayscale = np.mean(meanRGB)
        scale = grayscale / meanRGB
        white_balanced = np.zeros(image.shape)
        white_balanced[:, :, 0] = scale[2]*B
        white_balanced[:, :, 1] = scale[1]*G
        white_balanced[:, :, 2] = scale[0]*R
        return white_balanced

    def input1func(self, image=None):
        return self.white_balance(image=image)

    def input2func(self, image=None, gridsize=5):
        if image is None:
            return None
        # grayworld image
        bgr = np.uint8(self.white_balance(image=image))
        # lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB) Check
        lab = bgr
        l, a, b = (lab[:, :, 0], lab[:, :, 1], lab[:, :, 2])
        clahe = cv2.createCLAHE(
            clipLimit=3.5, tileGridSize=(gridsize, gridsize))
        l = clahe.apply(l)
        a = clahe.apply(a)
        b = clahe.apply(b)
        lab = np.uint8(np.dstack((l, a, b)))
        # final_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) Check
        final_img = lab
        return final_img

    def local_contrast_weight_map(self, image=None):
        """
        img: contrast enhanced RGB input image
        """
        if image is None:
            return None
        b, g, r = (image[:, :, 0].astype(np.float64), image[:, :, 1].astype(
            np.float64), image[:, :, 2].astype(np.float64))
        luminance = 0.2126*r+0.7152*g+0.0722*b
        luminance = (255*(luminance-np.min(luminance))) / \
            (np.max(luminance)-np.min(luminance))
        luminance_whc = cv2.GaussianBlur(luminance, (5, 5), 0)
        luminance_whc = (255*(luminance_whc-np.min(luminance_whc))) / \
            (np.max(luminance_whc)-np.min(luminance_whc))
        ret = np.abs(luminance - luminance_whc)
        ret = ret/255.0
        return ret

    def exposition_weight_map(self, image=None):
        """img: B/W?"""
        if image is None:
            return None
        b, g, r = (image[:, :, 0].astype(np.float64), image[:, :, 1].astype(
            np.float64), image[:, :, 2].astype(np.float64))
        img_gray = 0.2126*r+0.7152*g+0.0722*b
        img_gray = img_gray/255.0
        sigma = 0.5
        ret = np.exp(-(((img_gray-0.5)**2)/(2*sigma**2)))
        return ret

    def image_pyramid(self, image=None, pyramid_type='gaussian', levels=1):
        '''Function to generate the Gaussian/Laplacian pyramid of an image'''
        # Validate parameters
        if image is None:
            return None
        image = np.float64(image)
        # Generate Gaussian Pyramid
        current_layer = image
        gaussian = [current_layer]
        for i in range(levels):
            current_layer = cv2.pyrDown(current_layer)
            gaussian.append(current_layer)
        if pyramid_type == 'gaussian':
            return gaussian
        # Generate Laplacian Pyramid
        elif pyramid_type == 'laplacian':
            current_layer = gaussian[levels-1]
            laplacian = [current_layer]
            for i in range(levels-1, 0, -1):
                shape = (gaussian[i-1].shape[1], gaussian[i-1].shape[0])
                expand_gaussian = cv2.pyrUp(gaussian[i], dstsize=shape)
                current_layer = cv2.subtract(gaussian[i-1], expand_gaussian)
                laplacian.append(current_layer)
            laplacian.reverse()
            return laplacian

    def __clip(self, image=None):
        '''Function to clip images to range of [0.0, 1.0]'''
        # Validate parameters
        if image is None:
            return None
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    def fusion(self, inputs=None, weights=None, gaussians=None):
        '''Function to fuse the pyramids together'''
        # Validate parameters
        if inputs is None or weights is None or gaussians is None:
            return None
        fused_levels = []
        # Perform Fusion by combining the Laplacian and Gaussian pyramids
        for i in range(len(gaussians[0])):
            if len(inputs[0].shape) > 2:
                for j in range(inputs[0].shape[2]):
                    # Generate Laplacian Pyramids
                    laplacians = [
                        self.image_pyramid(
                            image=inputs[0][:, :, j], pyramid_type='laplacian', levels=len(gaussians[0])),
                        self.image_pyramid(
                            image=inputs[1][:, :, j], pyramid_type='laplacian', levels=len(gaussians[0]))
                    ]
                    # Adjust rows to match
                    row_size = np.min(np.array([
                        laplacians[0][i].shape[0],
                        laplacians[1][i].shape[0],
                        gaussians[0][i].shape[0],
                        gaussians[1][i].shape[0]
                    ]))
                    # Adjust columns to match
                    col_size = np.min(np.array([
                        laplacians[0][i].shape[1],
                        laplacians[1][i].shape[1],
                        gaussians[0][i].shape[1],
                        gaussians[1][i].shape[1]
                    ]))
                    if j == 0:
                        intermediate = np.ones(
                            inputs[0][:row_size, :col_size].shape)
                    # Fusion Step
                    intermediate[:, :, j] = (laplacians[0][i][:row_size, :col_size] * gaussians[0][i][:row_size, :col_size]) + (
                        laplacians[1][i][:row_size, :col_size] * gaussians[1][i][:row_size, :col_size])
            fused_levels.append(intermediate)

        # Reconstruct Image Pyramids
        for i in range(len(fused_levels)-2, -1, -1):
            level_1 = cv2.pyrUp(fused_levels[i+1])
            level_2 = fused_levels[i]
            r = min(level_1.shape[0], level_2.shape[0])
            c = min(level_1.shape[1], level_2.shape[1])
            fused_levels[i] = level_1[:r, :c] + level_2[:r, :c]

        # Clip fused image to [0.0, 1.0]
        fused = self.__clip(fused_levels[0])
        return fused

    def _assert_image_shapes_equal(self, org_img: np.ndarray, pred_img: np.ndarray, metric: str):
        assert org_img.shape == pred_img.shape

    def psnr(self, org_img: np.ndarray, pred_img: np.ndarray, max_p=255) -> float:
        self._assert_image_shapes_equal(org_img, pred_img, "PSNR")
        org_img = np.float64(org_img)
        mse_bands = []
        for i in range(org_img.shape[2]):
            mse_bands.append(
                np.mean(np.square(org_img[:, :, i] - pred_img[:, :, i])))
        return 20*(np.log10(max_p)) + (-10.*np.log10(np.mean(mse_bands)))

    def ssim(self, org_img: np.ndarray, pred_img: np.ndarray, max_p=255) -> float:
        self._assert_image_shapes_equal(org_img, pred_img, "SSIM")
        return structural_similarity(org_img, pred_img, data_range=max_p, multichannel=True)

    def main_control(self):
        self.input1 = self.input1func(image=self.img1)
        self.input2 = self.input2func(image=self.img1)
        self.input1normalised = self.input1/255.0
        self.input2normalised = self.input2/255.0
        self.weights = [
            # Weight maps for first image
            {
                'luminance': self.local_contrast_weight_map(image=self.input1),
                'chromatic': self.exposition_weight_map(image=self.input1)
            },
            # Weight maps for second image
            {
                'luminance': self.local_contrast_weight_map(image=self.input2),
                'chromatic': self.exposition_weight_map(image=self.input2)
            }
        ]
        self.weights[0]['combined'] = (
            self.weights[0]['luminance'] + self.weights[0]['chromatic'])
        self.weights[1]['combined'] = (
            self.weights[1]['luminance'] + self.weights[1]['chromatic'])
        self.weights[0]['normalized'] = self.weights[0]['combined'] / \
            (self.weights[0]['combined'] + self.weights[1]['combined'])
        self.weights[1]['normalized'] = self.weights[1]['combined'] / \
            (self.weights[0]['combined'] + self.weights[1]['combined'])
        self.input_images = [self.input1normalised, self.input2normalised]
        gaussians = [
            self.image_pyramid(
                image=self.weights[0]['normalized'], pyramid_type='gaussian', levels=self.pyramid_height),
            self.image_pyramid(
                image=self.weights[1]['normalized'], pyramid_type='gaussian', levels=self.pyramid_height)
        ]
        self.fused = self.fusion(self.input_images, self.weights, gaussians)
        return self.fused

    def run_code(self):
        fume_images = os.listdir(self.path_fume)
        ground_truth = os.listdir(self.path_ground_truth)
        # Images :
        # print("fume_images : ",fume_images)
        # print("ground_truth : ",ground_truth)
        if len(fume_images) != len(ground_truth):
            print(bcolor.WARNING+"Error :"+bcolor.ENDC,
                  "Number of Affected Images Failed to match with number of Ground Truth images")
            return None
        psnr_temp = []
        ssim_temp = []
        original_images = []
        predicted_images = []
        ground_truth_images = []
        for i in tqdm(range(len(fume_images))):
            self.img1 = cv2.imread(os.path.join(
                self.path_fume, fume_images[i]))
            self.img2 = np.float64(cv2.imread(os.path.join(
                self.path_ground_truth, ground_truth[i])))/255.0
            fused = self.main_control()
            original_images.append(self.img1)
            predicted_images.append(fused)
            ground_truth_images.append(self.img2)
            psnr_temp.append(self.psnr(org_img=np.uint8(
                self.img2*255), pred_img=np.uint8(fused*255)))
            ssim_temp.append(self.ssim(org_img=np.uint8(
                self.img2*255), pred_img=np.uint8(fused*255)))
        self.ssim = ssim_temp
        self.psnr = psnr_temp
        print(self.color.HEADER+"Some Similarity Matrices"+self.color.ENDC)
        print(self.color.OKCYAN+"SSIM :\n"+self.color.ENDC, self.ssim)
        print(self.color.OKCYAN+"PSNR :\n"+self.color.ENDC, self.psnr)
        print(self.color.OKCYAN+"SSIM Average :" +
              self.color.ENDC, np.mean(np.array(self.ssim)))
        print(self.color.OKCYAN+"PSNR Average :" +
              self.color.ENDC, np.mean(np.array(self.psnr)))
        return [original_images, predicted_images, ground_truth_images]

if __name__ == '__main__':
    bcolor = bcolors()
    if(len(sys.argv) != 3):
        print(bcolor.WARNING+"Command :"+bcolor.ENDC,
              "python3 <location folder of fume affected images> <location folder containing ground truth>")
    else:
        welding = weld_removal(sys.argv[1], sys.argv[2], bcolor)
        images = welding.run_code()
        _, axarr = plt.subplots(len(images[0]), len(images), figsize=[
                                len(images)*7, len(images[0])*30/4])
        title = ['Original Images', 'Our Outputs', 'Ground Truths']
        for i in range(len(images[0])):
            for j in range(len(images)):
                temp = images[j][i][:, :, ::-1]
                axarr[i][j].set_title(title[j])
                axarr[i][j].imshow(temp)
        plt.savefig("./figure.png")

