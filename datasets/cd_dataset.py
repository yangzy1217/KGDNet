import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from torchvision.transforms import functional as F


num_classes = 1


DATASET_CONFIGS = {
    'LEVIR-CD-256': {
        'root': '/home/dgxuser/Students/YZY/Dataset/LEVIR-CD-256/',
        'mean': np.array([101.47126742, 100.07854925, 85.45308413]),
        'std':  np.array([50.16481182, 47.89304896, 44.14429425]),
    },
    'LEVIR-CD+-256': {
        'root': '/raid/YZY/Dataset/LEVIR-CD+-256/',
        'mean': np.array([100.47507148,  99.10638696, 84.58167848]),
        'std':  np.array([50.16232948, 47.71950174, 43.87627161]),
    },
    'SYSU-CD': {
        'root': '/raid/YZY/Dataset/SYSU-CD/',
        'mean': np.array([101.82506142, 129.57220493, 110.21495806]),
        'std':  np.array([58.96176408, 46.84438154, 46.61458525]),
    },
    'WHU-CD': {
        'root': '/home/dgxuser/Students/YZY/Dataset/WHU-CD/',
        'mean': np.array([124.035, 118.824, 109.393]),
        'std':  np.array([50.378, 49.072, 52.329]),
    },
}


def get_root(dataset):
    return DATASET_CONFIGS[dataset]['root']


def normalize_image(im):
    im = im / 255
    return im.astype(np.float32)


def normalize_depth(im):
    im = im / 255
    return im.astype(np.float32)


def Color2Index(ColorLabel):
    return ColorLabel.clip(max=1)


def Index2Color(pred):
    return exposure.rescale_intensity(pred, out_range=np.uint8)


def sliding_crop_CD(imgs1, imgs2, depths1, depths2, labels, size):
    crop_imgs1, crop_imgs2 = [], []
    crop_depths1, crop_depths2 = [], []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img1, img2, depth1, depth2, label in zip(imgs1, imgs2, depths1, depths2, labels):
        h, w = img1.shape[:2]
        c_h, c_w = size
        if h < c_h or w < c_w:
            print(f"Cannot crop area {size} from image with size ({h}, {w})")
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_depths1.append(depth1)
            crop_depths2.append(depth2)
            crop_labels.append(label)
            continue
        h_rate = h / c_h
        w_rate = w / c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = 0 if h_times == 1 else math.ceil(c_h * (h_times - h_rate) / (h_times - 1))
        stride_w = 0 if w_times == 1 else math.ceil(c_w * (w_times - w_rate) / (w_times - 1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j * c_h - j * stride_h)
                s_h = h - c_h if j == (h_times - 1) else s_h
                e_h = s_h + c_h
                s_w = int(i * c_w - i * stride_w)
                s_w = w - c_w if i == (w_times - 1) else s_w
                e_w = s_w + c_w
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                crop_depths1.append(depth1[s_h:e_h, s_w:e_w, :])
                crop_depths2.append(depth2[s_h:e_h, s_w:e_w, :])
                if label_dims == 2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])
    print(f'Sliding crop finished. {len(crop_imgs1)} pairs of images created.')
    return crop_imgs1, crop_imgs2, crop_depths1, crop_depths2, crop_labels


def rand_crop_CD(img1, img2, depth1, depth2, label, size):
    h, w = img1.shape[:2]
    c_h, c_w = size
    if h < c_h or w < c_w:
        print(f"Cannot crop area {size} from image with size ({h}, {w})")
        return img1, img2, depth1, depth2, label
    s_h = random.randint(0, h - c_h)
    e_h = s_h + c_h
    s_w = random.randint(0, w - c_w)
    e_w = s_w + c_w
    return (
        img1[s_h:e_h, s_w:e_w, :],
        img2[s_h:e_h, s_w:e_w, :],
        depth1[s_h:e_h, s_w:e_w, :],
        depth2[s_h:e_h, s_w:e_w, :],
        label[s_h:e_h, s_w:e_w],
    )


def rand_flip_CD(img1, img2, depth1, depth2, label):
    r = random.random()
    if r < 0.25:
        return img1, img2, depth1, depth2, label
    elif r < 0.5:
        return (np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(),
                np.flip(depth1, axis=0).copy(), np.flip(depth2, axis=0).copy(),
                np.flip(label, axis=0).copy())
    elif r < 0.75:
        return (np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(),
                np.flip(depth1, axis=1).copy(), np.flip(depth2, axis=1).copy(),
                np.flip(label, axis=1).copy())
    else:
        return (img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(),
                depth1[::-1, ::-1, :].copy(), depth2[::-1, ::-1, :].copy(),
                label[::-1, ::-1].copy())


def rand_temporal_flip(img1, img2, depth1, depth2, label):
    if random.random() < 0.5:
        return img2, img1, depth2, depth1, label
    return img1, img2, depth1, depth2, label


def read_RSimages(root, mode, read_list=False):
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    depth_A_dir = os.path.join(root, mode, 'A_depth')
    depth_B_dir = os.path.join(root, mode, 'B_depth')
    label_dir = os.path.join(root, mode, 'label')
    if mode == 'train' and read_list:
        list_path = os.path.join(root, mode + '_info.txt')
        with open(list_path, 'r') as f:
            data_list = [line.strip() for line in f.readlines()]
    else:
        data_list = os.listdir(img_A_dir)
    data_A, data_B, depths_A, depths_B, labels = [], [], [], [], []
    for idx, it in enumerate(data_list):
        if it.endswith('.png'):
            img_A = normalize_image(io.imread(os.path.join(img_A_dir, it)))
            img_B = normalize_image(io.imread(os.path.join(img_B_dir, it)))
            label = Color2Index(io.imread(os.path.join(label_dir, it)))
            depth_A = normalize_depth(io.imread(os.path.join(depth_A_dir, it)))
            depth_B = normalize_depth(io.imread(os.path.join(depth_B_dir, it)))
            data_A.append(img_A)
            data_B.append(img_B)
            depths_A.append(depth_A)
            depths_B.append(depth_B)
            labels.append(label)
        if not idx % 50:
            print(f'{idx}/{len(data_list)} images loaded.')
    print(data_A[0].shape)
    print(f'{len(data_A)} {mode} images loaded.')
    return data_A, data_B, depths_A, depths_B, labels


class RS(data.Dataset):
    """Unified bi-temporal change-detection dataset with monocular depth.

    Supports LEVIR-CD-256, LEVIR-CD+-256, SYSU-CD, WHU-CD via DATASET_CONFIGS.
    All images are preloaded into RAM at construction time.
    """

    def __init__(self, dataset, mode, random_crop=False, crop_nums=6,
                 sliding_crop=False, crop_size=512,
                 random_flip=False, random_temporal_flip=False):
        if dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unsupported dataset: {dataset}. "
                f"Choices: {list(DATASET_CONFIGS)}"
            )
        self.dataset = dataset
        self.root = DATASET_CONFIGS[dataset]['root']

        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        self.random_temporal_flip = random_temporal_flip

        data_A, data_B, depths_A, depths_B, labels = read_RSimages(
            self.root, mode, read_list=False
        )
        if sliding_crop:
            data_A, data_B, depths_A, depths_B, labels = sliding_crop_CD(
                data_A, data_B, depths_A, depths_B, labels,
                [self.crop_size, self.crop_size]
            )
        self.data_A, self.data_B = data_A, data_B
        self.depths_A, self.depths_B = depths_A, depths_B
        self.labels = labels
        self.len = crop_nums * len(data_A) if self.random_crop else len(data_A)

    def __getitem__(self, idx):
        if self.random_crop:
            orig_idx = idx // self.crop_nums
            data_A = self.data_A[orig_idx]
            data_B = self.data_B[orig_idx]
            depth_A = self.depths_A[orig_idx]
            depth_B = self.depths_B[orig_idx]
            label = self.labels[orig_idx]
            data_A, data_B, depth_A, depth_B, label = rand_crop_CD(
                data_A, data_B, depth_A, depth_B, label,
                [self.crop_size, self.crop_size]
            )
        else:
            data_A = self.data_A[idx]
            data_B = self.data_B[idx]
            depth_A = self.depths_A[idx]
            depth_B = self.depths_B[idx]
            label = self.labels[idx]

        if self.random_temporal_flip:
            data_A, data_B, depth_A, depth_B, label = rand_temporal_flip(
                data_A, data_B, depth_A, depth_B, label
            )
        if self.random_flip:
            data_A, data_B, depth_A, depth_B, label = rand_flip_CD(
                data_A, data_B, depth_A, depth_B, label
            )
        return (F.to_tensor(data_A), F.to_tensor(data_B),
                F.to_tensor(depth_A), F.to_tensor(depth_B), label)

    def __len__(self):
        return self.len
