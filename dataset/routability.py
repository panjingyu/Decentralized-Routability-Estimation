"""Modules for routability dataset."""


import os

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class DRVHotspotDataset(Dataset):
    """DRV hotspot detection dataset."""

    def __init__(self, root, train=True, n_channels=64, label_size=224,
                 transform=None):
        file_base = 'train' if train else 'val'
        if label_size == 224:
            label_base = ''
        else:
            label_base = '.{0}x{0}'.format(label_size)
        if n_channels == 64:
            feature_base = ''
        else:
            feature_base = '.{}ch'.format(n_channels)
        if isinstance(root, str):
            self.features_path = os.path.join(
                root, 'features{}/{}.npy'.format(
                    feature_base, file_base))
            self.labels_path = os.path.join(
                root, 'labels.drv-hotspot{}/{}.npy'.format(
                    label_base, file_base))
            self.features = np.load(self.features_path, mmap_mode='c')
            self.labels = np.load(self.labels_path)
        else:
            self.features_path = sorted([
                os.path.join(r, 'features{}/{}.npy'.format(
                    feature_base, file_base))
                for r in root
            ])
            self.labels_path = sorted([
                os.path.join(r, 'labels.drv-hotspot{}/{}.npy'.format(
                    label_base, file_base))
                for r in root
            ])
            self.features = np.vstack([np.load(p, mmap_mode='c')
                                       for p in self.features_path])
            self.labels = np.vstack([np.load(p) for p in self.labels_path])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.5,) * n_channels,
                                     (.5,) * n_channels),
            ])
        else:
            self.transform = transform
        self.n_channels = n_channels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, self.labels[idx]


class ViolatedNetRatioDataset(Dataset):
    """Violated net ratio dataset."""

    def __init__(self, root, train=True, n_channels=64, transform=None):
        file_base = 'train' if train else 'val'
        if isinstance(root, str):
            self.features_path = os.path.join(root,
                                            'features/{}.npy'.format(file_base))
            self.labels_path = os.path.join(root,
                                            'labels.violated-net-ratio/{}.npy'\
                                                .format(file_base))
            self.features = np.load(self.features_path, mmap_mode='c')
            self.labels = np.load(self.labels_path)
        else:
            self.features_path = sorted([
                os.path.join(r, 'features/{}.npy'.format(file_base))
                for r in root
            ])
            self.labels_path = sorted([
                os.path.join(
                    r,
                    'labels.violated-net-ratio/{}.npy'.format((file_base)))
                for r in root
            ])
            self.features = np.vstack([np.load(p, mmap_mode='c')
                                       for p in self.features_path])
            self.labels = np.hstack([np.load(p) for p in self.labels_path])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.5,) * n_channels,
                                     (.5,) * n_channels),
            ])
        else:
            self.transform = transform
        self.n_channels = n_channels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        #ch17 = [0, 1, 2, 6, 18, 19, 23, 24, 43, 44, 48, 49, 53, 54, 58, 59, -1]
        # feature = np.dstack(
        #     [feature[..., ch] for ch in ch17])
        if self.transform:
            feature = self.transform(feature)
        return feature, self.labels[idx]

