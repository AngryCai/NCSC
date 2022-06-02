import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from torch.utils.data import Dataset

from Toolbox.Preprocessing import Processor
from utils.superpixel_utils import HSI_to_superpixels, create_association_mat, create_spixel_graph


class HSI_Data(Dataset):
    def __init__(self, path_to_data, path_to_gt, path_to_sp=None, patch_size=(7, 7), num_superpixel=500, transform=None, pca=True,
                 pca_dim=8, is_superpixel=True, is_labeled=True):
        self.transform = transform
        self.num_superpixel = num_superpixel
        self.path_to_sp = path_to_sp
        p = Processor()
        img, gt = p.prepare_data(path_to_data, path_to_gt)
        self.img = img
        self.gt = gt
        n_row, n_column, n_band = img.shape
        if pca:
            img = scale(img.reshape(n_row * n_column, -1))  # .reshape((n_row, n_column, -1))
            pca = PCA(n_components=pca_dim)
            img = pca.fit_transform(img).reshape((n_row, n_column, -1))
        if is_superpixel:
            if path_to_sp is not None:
                self.sp_labels = loadmat(self.path_to_sp)['labels']
                # show_superpixel(self.sp_labels, img[:, :, :3])
            else:
                self.sp_labels = HSI_to_superpixels(img, num_superpixel=self.num_superpixel, is_pca=False,
                                                    is_show_superpixel=False)
                # show_superpixel(self.sp_labels, img[:, :, :3])
            self.association_mat = create_association_mat(self.sp_labels)
            self.sp_graph, self.sp_centroid = create_spixel_graph(img, self.sp_labels)
        x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
        y = p.standardize_label(y_)
        for i in np.unique(y):
            print(np.nonzero(y == i)[0].shape[0])

        if not is_labeled:
            self.n_classes = np.unique(y).shape[0] - 1
        else:
            self.n_classes = np.unique(y).shape[0]
        n_samples, n_row, n_col, n_channel = x_patches.shape
        self.data_size = n_samples
        x_patches = scale(x_patches.reshape((n_samples, -1))).reshape((n_samples, n_row, n_col, -1))
        x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
        self.x_tensor, self.y_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor), \
                                       torch.from_numpy(y).type(torch.LongTensor)

    def __getitem__(self, idx):
        x, y = self.x_tensor[idx], self.y_tensor[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.data_size
