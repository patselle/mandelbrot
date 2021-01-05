import os
import sys
import fnmatch
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import pywt

import matplotlib.pyplot as plt



class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(f"readable_dir:{prospective_dir} is not a valid path")
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest,  prospective_dir)
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{prospective_dir} is not a readable path")


class writeable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(f"writeable_dir:{prospective_dir} is not a valid path")
        if os.access(prospective_dir, os.W_OK):
            setattr(namespace, self.dest,  prospective_dir)
        else:
            raise argparse.ArgumentTypeError(f"writeable_dir:{prospective_dir} is not a writeable path")


class converter:
    def __init__(self, input_root, output_root, tmp_root, rgb2gray, wavelet, window_size, x_step, y_step, takeABS, output_type) -> None:
        self.__input_root = input_root
        self.__output_root = output_root
        self.__tmp_root = tmp_root
        self.__rgb2gray: bool = rgb2gray
        self.__window_size = window_size # (6,8)
        self.__x_step: int = x_step
        self.__y_step: int = y_step
        self.__takeABS: bool = takeABS
        self.__output_type: str = output_type

        self.__height: int = None
        self.__width: int = None

        self.__df = None
        self.__label = None

        self.__wavelet: str = wavelet


    def __iterate_files(self, dirpath, pattern = None):
        if pattern is None:
            pattern = '*'

        for root, dirs, files in os.walk(dirpath, topdown=False):
            for file in files:
                if fnmatch.fnmatch(file, pattern):
                    yield os.path.relpath(os.path.join(root, file), dirpath)


    def __load_image(self, input_path):
        img_array = cv2.imread(input_path)
        img_array = img_array[...,::-1]

        # if image is aleady a gray scale image return img_array
        if self.__rgb2gray:
            return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            return img_array


    def __do_wavelet(self, img):
        # returns LL_coeffs, (LH_coeffs, HL_coeffs, HH_coeffs)
        # ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
        coeffs = pywt.dwt2(img, self.__wavelet)
        return coeffs


    def __create_heatmap(self, coeffs):
        # start from the top left edge
        idx_x = 0
        idx_y = 0

        patchies = []

        # create a list of patchies 
        while(True):
            # when the bottem end is reached, start from the top with a y-step.  
            if idx_x + self.__window_size[0] > coeffs.shape[0]:
                idx_x = 0
                idx_y += self.__y_step

                # if the right image border is exceeded, cancel
                if (idx_y + self.__window_size[1] > coeffs.shape[1]):
                    break

            patchies.append(coeffs[idx_x: idx_x + self.__window_size[0], idx_y: idx_y + self.__window_size[1]])            
            idx_x += self.__x_step
        
        # create heatmap via mean pooling
        heatmap = np.ones(len(patchies))
        for i in range(heatmap.shape[0]):
            heatmap[i] = patchies[i].mean()

        # reshape heatmap from 1D to 2D and transpose it
        size_x = int(1. / self.__x_step * (coeffs.shape[0] - self.__window_size[0]) + 1)
        size_y = int(1. / self.__y_step * (coeffs.shape[1] - self.__window_size[1]) + 1) 

        heatmap = np.abs(heatmap).reshape(size_x, size_y).T

        # resize heatmap back to the original image
        heatmap_resized = np.abs(cv2.resize(heatmap, (self.__width, self.__height), interpolation=cv2.INTER_CUBIC))
        heatmap_resized = heatmap_resized.astype(np.uint8)

        return heatmap_resized


    def __get_RoI(self, heatmap):
        # get pixels of RoI
        label = np.unravel_index(heatmap.argmax(), heatmap.shape)

        if self.__output_type == 'probability':
            percent_ax0 = 100.0 / self.__height * label[0]
            percent_ax1 = 100.0 / self.__width * label[1]
        elif self.__output_type == 'pixel':
            percent_ax0 = label[0]
            percent_ax1 = label[1]
        else:
            print('wrong self.__output_type exit')
            sys.exit()

        self.__label = (percent_ax0, percent_ax1)


    def __write_csv(self, counter, file):


        child_dir = os.path.basename(file).split('.')[0]
        parent_dir = os.path.dirname(file)

        self.__df.loc[[counter], 'file'] =  f'{parent_dir}_{child_dir}'
        self.__df.loc[[counter], 'dim0'] =  self.__label[0]
        self.__df.loc[[counter], 'dim1'] =  self.__label[1]


    def __save_csv(self):
        self.__df.to_csv(os.path.join(self.__output_root, 'labels.csv'), sep=';', index=False)
    

    def run(self):
        # open csv file and recreate it if neccessary
        csv_file = os.path.join(self.__output_root, 'label.csv')
        if os.path.isfile(csv_file):
            print('csv found file')
            self.__df = pd.read_csv(csv_file)
        else:
            np_arr = np.zeros(shape=(len([_ for _ in self.__iterate_files(self.__input_root)]), 3))
            self.__df = pd.DataFrame(np_arr, columns=['file', 'dim0', 'dim1'])
            print('No csv found file, create new one') 

        for i, f in enumerate(self.__iterate_files(self.__input_root)):
            
            input_path = os.path.join(self.__input_root, f)

            # load image via cv2 and convert to gray if needed
            img_array = self.__load_image(input_path)
            self.__height, self.__width = img_array.shape[0], img_array.shape[1]

            # do wavelet, only consider HH coeffs
            _, (_, _, HH_coeffs) = self.__do_wavelet(img_array)

            # take absolute value, a different but also interesting point will maybe chosen
            if self.__takeABS:
                HH_coeffs = abs(HH_coeffs)

            # do patchy analysis
            heatmap = self.__create_heatmap(HH_coeffs)

            # save temporary heatmap files
            if self.__tmp_root:
                print(f'saving tmp: {f}')
                im = Image.fromarray(heatmap)
                im.save(os.path.join(self.__tmp_root, f))

            # compute RoI and extract coordiates
            self.__get_RoI(heatmap)

            # save label to our label file
            self.__write_csv(i, f)
        
        self.__save_csv()


def get_argument_parser():
    parser = argparse.ArgumentParser(description='main')

    parser.add_argument('--input_root', '-i', action=readable_dir, default=os.path.join(os.getcwd(), 'input'), help='Path to root dir of source files')
    parser.add_argument('--output_root', '-o', action=writeable_dir, default=os.path.join(os.getcwd(), 'output'), help='Path to root dir of output files')
    parser.add_argument('--tmp_root', '-p', action=writeable_dir, default=None, help='Path to root dir of tmp files')
    parser.add_argument('--rgb2gray', '-t',type=bool, default=False, help='Convert RGB image to gray scale image')
    parser.add_argument('--wavelet', '-w',type=str, default='bior1.3', help='Mother wavelet, default bior1.3, maybe try haar')
    parser.add_argument('--window_size', '-s',type=str, nargs='+', default=(6,8), help='Patch size, default (6,8)')
    parser.add_argument('--x_step', '-x',type=int, default=3, help='x step size, default 3')
    parser.add_argument('--y_step', '-y',type=int, default=4, help='y step size, default 4')
    parser.add_argument('--takeABS', type=bool, default=False, help='Take the absolute value of the wavelet transformed, default: false')
    parser.add_argument('--output_type', type=str, default=False, help='Specify coordinate format, choose: pixel or probability, default: pixel')

    return parser.parse_args()



if __name__ == "__main__":
    args_dict = vars(get_argument_parser())

    m_converter = converter(**args_dict)
    m_converter.run()
