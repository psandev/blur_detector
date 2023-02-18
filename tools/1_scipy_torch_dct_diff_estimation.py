'''
estimate the differences between the official dct and the torch implenetation
'''

import torch
import matplotlib
import numpy as np
from scipy.fft import dct
import torch_dct as torch_dct
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img = np.random.randint(255, size=[64, 64, 3]).astype(np.float32)
    img = img[..., 0] /4 + img[..., 1]/2 + img[..., 2]/4  # grayscale conversion
    res = dct(img)
    torch_res = torch_dct.dct(torch.from_numpy(img)).numpy()

    arr_diff = res - torch_res
    diff_max, dif_min = arr_diff.max(), arr_diff.min()
    print(f'{diff_max = :e}, {dif_min= :e}')

    fig, ax = plt.subplots()
    ax.imshow(res.astype(np.uint8))
    ax.axis('off')
    plt.show(block=False)

    fig1, ax1 = plt.subplots()
    ax1.imshow(torch_res.astype(np.uint8))
    ax1.axis('off')
    plt.show()




