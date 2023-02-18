"""
Find the right blur types and their parameters in order to create the dataset

This nice little library has all that is needed for this task.
https://github.com/NatLee/Blur-Generator
"""
import random

import cv2
from random import choices
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from wcmatch.pathlib import Path
from blurgenerator import motion_blur, lens_blur, gaussian_blur


if __name__ == '__main__':
    FOLDER = '/data1/riverside/faces_split'
    IM_SIZE = (416, 416)
    SEED = 42
    NIMAGES = 1
    REPS = 5
    TITLE_FONTSIZE = 10



    path = Path(FOLDER)
    images = tuple(path.rglob(['*.png', '*.jpeg', '*.jpg']))

    # random.seed(SEED)
    rand_idx = choices(range(len(images)), k=NIMAGES)
    images = [images[x] for x in rand_idx]
    for image in images:
        image = cv2.cvtColor(cv2.imread(image.as_posix()), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IM_SIZE, cv2.INTER_LANCZOS4)

        # Motion blur - change size and keep angle constant
        fig1, ax1 = plt.subplots(1,REPS)
        fig1.canvas.manager.set_window_title('Motion Blur')
        for i, size in enumerate(range(1, 10, 10//REPS)):
            img_blur = motion_blur(image, size=size, angle=30)
            ax1[i].imshow(img_blur)
            ax1[i].axis('off')
            ax1[i].set_title(f'{size=}, angle=30', fontsize=TITLE_FONTSIZE)
        fig1.tight_layout()
        plt.show(block=False)
        # Gaussian blur - change kernel size and keep sigma constant
        fig2, ax2 = plt.subplots(2, REPS)
        fig2.canvas.manager.set_window_title('Gaussian Blur')
        for i, kernel in enumerate(range(3, 12, 12 // REPS)):
            img_blur = gaussian_blur(image, kernel=kernel)
            ax2[0, i].imshow(img_blur)
            ax2[0, i].axis('off')
            ax2[0, i].set_title(f'{kernel=}, sigma=5',fontsize=TITLE_FONTSIZE)

        # Gaussian blur - change sigma size and keep kernel size constant
        for i, sigma in enumerate(range(1, 20, 20 // REPS)):
            img_blur = gaussian_blur(image, kernel=5, sigma=sigma)
            ax2[1, i].imshow(img_blur)
            ax2[1, i].axis('off')
            ax2[1, i].set_title(f'kernel=5, {sigma=}', fontsize=TITLE_FONTSIZE)
        fig2.tight_layout()
        plt.show(block=False)

        # Lens blur - change radius and keep components and exposure_gama constant
        fig3, ax3 = plt.subplots(3, REPS)
        fig3.canvas.manager.set_window_title('Lens Blur')
        for i, radius in enumerate(range(3, 12, 12 // REPS)):
            img_blur = lens_blur(image, radius=radius)
            ax3[0, i].imshow(img_blur)
            ax3[0, i].axis('off')
            ax3[0, i].set_title(f'{radius=}, components=5, exposure_gamma=5', fontsize=TITLE_FONTSIZE)

        # Lens blur - change components and keep kernel and exposure_gama constant
        for i, components in enumerate(range(1, 6)):
            img_blur = lens_blur(image, components=components)
            ax3[1, i].imshow(img_blur)
            ax3[1, i].axis('off')
            ax3[1, i].set_title(f'kernel=5, {components=}, exposure_gamma=5', fontsize=TITLE_FONTSIZE)



        # Lens blur - change exposure_gama and keep kernel and components constant
        for i, exposure_gamma in enumerate(range(2, 7)):
            img_blur = lens_blur(image, exposure_gamma=exposure_gamma)
            ax3[2, i].imshow(img_blur)
            ax3[2, i].axis('off')
            ax3[2, i].set_title(f'kernel=5, components, {exposure_gamma=}', fontsize=TITLE_FONTSIZE)
        fig3.tight_layout()
        plt.show()


"""
Realistic parameters based on images review

motion blur:    radius = [3,5]
                angle = range(0, 360, 10
               
gaussian blur:  size = [3,5]
                sigma = range(1,9,2)
                
lense_blur:     radius = [3,5]
                componenets = range(1,6)
                exposure_gamma = range(2,7)

"""