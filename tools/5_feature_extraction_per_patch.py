import cv2
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy
from wcmatch.pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from multiprocessing.pool import Pool


"""
remove dct = 0, (dc part)
split ac part into 3 regions: low, medium and high frequencies
for each 8x8 pathch calculate: mean, std, kurtosis, skewness, entropy, and energy
Average per image
Al in all : 18 featured vector per image     
"""

def energy(img):
    if len(img) == 0:
        return np.nan

    # Compute X derivative of the image
    sobel_x = cv2.Sobel(img ,cv2.CV_64F, 1, 0, ksize=3)

    # Compute Y derivative of the image
    sobel_y = cv2.Sobel(img ,cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # Return weighted summation of the two images i.e. 0.5*X + 0.5*Y
    energy = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return energy.sum()/img.size



def process_row_pandas(x):
    # global patch_size
    res = []
    data = x.iloc[:patch_size]
    data_low_freq = data[(data>0) | (data>1300)]
    data_med_freq = data[(data > 1300) | (data > 2600)]
    data_high_freq = data[data > 2600]
    for df in [data_low_freq, data_med_freq, data_high_freq]:
        res_img = []
        if len(df) == 0:
            res_img.extend([np.nan] * 6)
        else:
            res_img.extend(df.agg(['mean', 'std', energy]).tolist())
            res_img.append(kurtosis(df))
            res_img.append(skew(df))
            res_img.append(shannon_entropy(df))
        res.extend(res_img)
    res.append(x.img_num)
    res.append(x.train_val)
    res.append(x.label)
    return res

def process_row_multiptocessing(x):
    # global patch_size
    res = []
    data = x[:patch_size]
    data_low_freq = data[(data > 0) | (data > 1300)]
    data_med_freq = data[(data > 1300) | (data > 2600)]
    data_high_freq = data[data > 2600]
    for df in [data_low_freq, data_med_freq, data_high_freq]:
        res_img = []
        if len(df) == 0:
            res_img.extend([np.nan] * 6)
        else:
            res_img.append(df.mean())
            res_img.append(df.std())
            res_img.append(energy(df))
            res_img.append(kurtosis(df))
            res_img.append(skew(df))
            res_img.append(shannon_entropy(df))
        res.extend(res_img)
    res.append(x[-3])
    res.append(x[-2])
    res.append(x[-1])
    return res


if __name__ == '__main__':
    FOLDER = '/data1/riverside/blur_dataset_features'
    TRAIN_VAL_DICT = {'train': 0, 'val': 1}
    LABEL_DICT = {'blur': 1, 'no_blur': 0}
    path = Path(FOLDER)
    files = tuple(path.glob('*.npy'))
    df_features_list = []
    nimages = 0
    for file in tqdm(files, total=len(files)):
        stage, label = file.stem.split('_',1)

        data = np.load(file)
        df_list = []
        npatches = data.shape[1]
        patch_size = data.shape[2] * data.shape[3]
        for i in tqdm(range(data.shape[0]), total=nimages):
            df_im = pd.DataFrame(data[i].reshape(npatches, -1))
            df_im['img_num'] = nimages + i
            df_im['label'] = LABEL_DICT[label]
            df_im['train_val'] = TRAIN_VAL_DICT[stage]
            df_list.append(df_im)
        nimages += data.shape[0]
        df = pd.concat(df_list)


        pool = Pool(cpu_count())
        rows = df.values
        features = pool.map(process_row_multiptocessing, tqdm(rows))
        pool.close()
        pool.join()


        df_features_list.append(pd.DataFrame(features))
    df_all = pd.concat(df_features_list)
    df_all.columns = ['low_freq_mean', 'low_freq_std', 'low_freq_energy', 'low_freq_kurt', 'low_freq_skew', 'low_freq_entropy',
                      'med_freq_mean', 'med_freq_std', 'med_freq_energy', 'med_freq_kurt', 'med_freq_skew', 'med_freq_entropy',
                      'high_freq_mean', 'high_freq_std', 'high_freq_energy', 'high_freq_kurt', 'high_freq_skew', 'high_freq_entropy',] + [ 'img_num', 'label', 'train_val']
    df_all.to_parquet(path/'features.parquet', index=None)