"""
crete the image datset

2 blurs per image selected with replacement out of the 3
randomlly selected with replacemnt blur parameters out of the chosen below

motion blur:    radius = [3,5]
                angle = range(0, 360, 10

gaussian blur:  size = [3,5]
                sigma = range(1,9,2)

lense_blur:     radius = [3,5]
                componenets = range(1,6)
                exposure_gamma = range(2,7)

"""


import shutil
import cv2
import numpy as np
from wcmatch.pathlib import Path
from sklearn.model_selection import train_test_split
from random import randint
from tqdm import tqdm
from skimage import img_as_ubyte
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from blurgenerator import motion_blur, lens_blur, gaussian_blur



if __name__ == '__main__':
    FOLDER_IN = '/data1/riverside/faces_split'
    FOLDER_OUT = '/data1/riverside/blur_dataset'
    TEST_RATIO = 0.1
    LABEL_DICT = {0: 'no_blur', 1: 'blur'}
    SEED = 42
    REPS = 2 # blurred images per original image
    IMG_SIZE = (512, 512)

    # BLUR PARAMETERs PER TYPE
    MB_SIZE = (3, 5)
    MB_ANGLE = tuple(range(0, 360, 10))

    GB_KERNEL = (3, 5)
    GB_SIGMA = list(range(1, 9, 2))

    LB_RADIUS = (3, 5)
    LB_COMPTS = tuple(range(1, 6))
    LB_EXP_GAMMA = tuple(range(2, 7))


    blurs = {0:'mb' , 1:'gb', 2:'lb'}






    path_in = Path(FOLDER_IN)
    path_out = Path(FOLDER_OUT)
    if path_out.exists():
        shutil.rmtree(path_out)


    images = list(path_in.rglob(['*.png', '*.jpeg', '*.jpg']))
    train, val = train_test_split(images, random_state=42, test_size=TEST_RATIO)
    data_dict = {'train': train, 'val': val}
    for key, dataset in data_dict.items():
        path_no_blur = path_out/key/'no_blur'
        path_blur = path_out/key/'blur'
        path_no_blur.mkdir(parents=True, exist_ok=True)
        path_blur.mkdir(parents=True, exist_ok=True)

        for file in tqdm(dataset, total=len(dataset)):
            new_path_no_blur = path_no_blur/file.name

            # each time randomly select a blur and randomly choose its parameters
            for i in range(REPS):
                blur_type_inx = randint(0, len(blurs) - 1)

                img = cv2.imread(file.as_posix())
                h, w = img.shape[:-1]
                if not h == IMG_SIZE[0] or not w == IMG_SIZE[1]:
                    img = cv2.resize(img, IMG_SIZE, cv2.INTER_LANCZOS4)
                    cv2.imwrite(new_path_no_blur.as_posix(), img)
                else:
                    shutil.copy(file, new_path_no_blur)
                # img = cv2.normalize(img, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                if blur_type_inx == 0:
                    lb_size_idx = randint(0, len(MB_SIZE)-1)
                    lb_angle_idx = randint(0, len(MB_ANGLE)-1)
                    size = MB_SIZE[lb_size_idx]
                    angle = MB_ANGLE[lb_angle_idx]
                    img_blur = motion_blur(img, size=size, angle=angle)
                    new_file_name = f'{path_blur/file.stem}_{blurs[blur_type_inx]}_' \
                                    f'{size}_{angle}_{i}{file.suffix}'
                if blur_type_inx == 1:
                    gb_kernel_idx = randint(0, len(GB_KERNEL)-1)
                    gb_sigma_idx = randint(0, len(GB_SIGMA)-1)
                    kernel = GB_KERNEL[gb_kernel_idx]
                    sigma = GB_SIGMA[gb_sigma_idx]

                    img_blur = gaussian_blur(img, kernel=kernel, sigma=sigma)
                    new_file_name = f'{path_blur / file.stem}_{blurs[blur_type_inx]}_' \
                                    f'{kernel}_{sigma}_{i}{file.suffix}'
                if blur_type_inx == 2:
                    lb_radius_idx = randint(0, len(LB_RADIUS)-1)
                    lb_compts_idx = randint(0, len(LB_COMPTS)-1)
                    lb_exp_gamma_idx = randint(0, len(LB_EXP_GAMMA)-1)
                    radius = LB_RADIUS[lb_radius_idx]
                    components = LB_COMPTS[lb_compts_idx]
                    exp_gamma = LB_EXP_GAMMA[lb_exp_gamma_idx]
                    img_blur = lens_blur(img, radius=radius,
                                         components=components,
                                         exposure_gamma=exp_gamma)

                    new_file_name = f'{path_blur / file.stem}_{blurs[blur_type_inx]}_' \
                                    f'{radius}_{components}_{exp_gamma}_{i}{file.suffix}'

                img_blur = img_as_ubyte(img_blur)
                cv2.imwrite(new_file_name, img_blur)
