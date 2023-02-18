import torch
import shutil
import torch_dct
import kornia as K
import numpy as np
from wcmatch.pathlib import Path
from tqdm import tqdm

from kornia.contrib import extract_tensor_patches

from torchvision.io import read_image


if __name__ == '__main__':
    FOLDER_IN = '/data1/riverside/blur_dataset'
    FOLDER_OUT = '/data1/riverside/blur_dataset_features'
    PATCH_SIZE = 8  # PIXELS
    MEAN_RGB = (125.707, 109.054, 98.535)
    STD_RGB = (66.007, 60.502, 59.939)

    MEAN_GRAY = [110.588]
    STD_GRAY =  [60.293]

    train_blur, train_no_blur, val_blur, val_no_blur = [], [], [], []

    path_in = Path(FOLDER_IN)
    path_out = Path(FOLDER_OUT)
    if path_out.exists():
        shutil.rmtree(path_out)
    path_out.mkdir()

    images = tuple(path_in.rglob(['*.png', '*.jpeg', '*.jpg']))
    norm = K.augmentation.Normalize(mean=torch.tensor(MEAN_GRAY), std=torch.tensor(STD_GRAY))
    dsets = [train_blur,]
    for image in tqdm(images, total=len(images)):
        parts = image.parts
        stage, label = parts[-3], parts[-2]
        dset = eval(f'{stage}_{label}')
        arr = read_image(image.as_posix()).to(torch.float32).unsqueeze(0)
        arr = arr[:, 0, ...] / 4 + arr[:,1 , ...] / 2 + arr[:, 2, ...] / 4  # grayscale conversion
        # arr = norm(arr.unsqueeze(0))
        patches = extract_tensor_patches(arr.unsqueeze(0), window_size=PATCH_SIZE, stride=PATCH_SIZE)
        patches = patches.squeeze(0).squeeze(1)
        res_dcm = torch_dct.dct(patches)
        dset.append(res_dcm)

    for var in [train_blur, train_no_blur, val_blur, val_no_blur]:
        var_name = [k for k,v in locals().items() if v is var][0]
        np.save((path_out/var_name).with_suffix('.npy'), np.stack(var))






