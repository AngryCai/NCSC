import os
import numpy as np
import torch
import argparse

from sklearn.decomposition import PCA

from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import scale
from scipy.io import savemat
from sklearn.preprocessing import minmax_scale
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = "D:\\Python\\HSI_Files\\"
    dataset = 'HSI-TrT'
    # prepare data
    if dataset == "HSI-SaA":
        im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    elif dataset == "HSI-InP":
        im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    elif dataset == "HSI-InP-2010":
        im_, gt_ = 'Indian_Pines_2010', 'Indian_Pines_2010_gt'
    elif dataset == "HSI-SaN":
        im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    elif dataset == "HSI-PaC":
        im_, gt_ = 'Pavia', 'Pavia_gt'
    elif dataset == "HSI-Hou":
        im_, gt_ = 'Houston', 'Houston_gt'
    elif dataset == "HSI-TrT":
        im_, gt_ = 'Trento-HSI', 'Trento-GT'
    else:
        raise NotImplementedError
    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape

    img = scale(img.reshape(n_row * n_column, -1))  # .reshape((n_row, n_column, -1))
    pca = PCA(n_components=3)

    img = pca.fit_transform(img)
    img = minmax_scale(img)
    img = img.reshape((n_row, n_column, -1))
    img = ndi.gaussian_filter(img, 0.1)

    plt.imshow(img)
    plt.show()
    print('pca dim=', img.shape[-1])
    print('Processing %s ' % img_path)
    savemat('pca_img/{:s}-pca.mat'.format(dataset), {'img_pca': img})

    from PIL import Image
    # img_arr2pil = Image.fromarray(img)
    # im = Image.fromarray(255 * img.astype(np.uint8)).convert('RGB')
    # im.save('pca_img/{:s}-pca.jpg'.format(dataset))
    plt.imsave('pca_img/{:s}-pca.jpg'.format(dataset), img[:, :, :3])
