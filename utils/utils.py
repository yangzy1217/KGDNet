import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def read_idtxt(path):
    id_list = []
    f = open(path, 'r')
    curr_str = ''
    while True:
        ch = f.read(1)
        if is_number(ch):
            curr_str += ch
        else:
            id_list.append(curr_str)
            curr_str = ''
        if not ch:
            break
    f.close()
    return id_list


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape: (H, W, C)."""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yield lists by batch."""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def seprate_batch(dataset, batch_size):
    """Yield dataset items in batches."""
    num_batch = len(dataset) // batch_size + 1
    batch_len = batch_size
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        if i + 2 == num_batch:
            batch_len = len(dataset) - (num_batch - 1) * batch_size
    return batches


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# Credits to https://stackoverflow.com/users/6076729/manuel-lagunas.
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # Avoid issues with '1' at the start or end of the flattened mask.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


class AverageMeter(object):
    """Compute and store the average and current value."""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def ImageValStretch2D(img):
    img = img * 255
    return img.astype(int)


def ConfMap(output, pred):
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
        for w_idx in range(w):
            n_idx = int(pred[h_idx, w_idx])
            sum = 0
            for i in range(n):
                val = output[i, h_idx, w_idx]
                if val > 0:
                    sum += val
            conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx] / sum
            if conf[h_idx, w_idx] < 0:
                conf[h_idx, w_idx] = 0
    return conf


def accuracy(pred, label):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def align_dims(np_input, target_dim=2):
    """Adjust a NumPy array to the requested number of dimensions."""
    np_output = np_input
    while np_output.ndim > target_dim:
        for axis in range(np_output.ndim):
            if np_output.shape[axis] == 1:
                np_output = np_output.squeeze(axis)
                break
        else:
            break
    while np_output.ndim < target_dim:
        np_output = np.expand_dims(np_output, axis=0)
    return np_output


def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * label).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.999 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove detections from unlabeled ground-truth pixels.
    imPred = imPred * (imLab > 0)

    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass + 1))

    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass + 1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass + 1))
    area_union = area_pred + area_lab - area_intersection
    return area_intersection, area_union


def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove detections from unlabeled ground-truth pixels.
    imPred = imPred * (imLab > 0)

    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass + 1))

    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass + 1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass + 1))

    union_hist = pred_hist + lab_hist - TP_hist
    return TP_hist, pred_hist, lab_hist, union_hist


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = self.alpha * (1 - pt) ** self.gamma
        loss = focal_term * bce_loss
        return loss.mean()


def edge_loss(pred, target, eps=1e-8):
    pred_prob = torch.sigmoid(pred)
    sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]], device=pred.device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]], device=pred.device).view(1, 1, 3, 3)

    pred_edge_x = F.conv2d(pred_prob, sobel_kernel_x, padding=1)
    pred_edge_y = F.conv2d(pred_prob, sobel_kernel_y, padding=1)
    target_edge_x = F.conv2d(target, sobel_kernel_x, padding=1)
    target_edge_y = F.conv2d(target, sobel_kernel_y, padding=1)

    max_gradient = 4 * torch.sqrt(torch.tensor(2.0, device=pred.device))
    pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + eps) / max_gradient
    target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + eps) / max_gradient

    return F.l1_loss(pred_edge, target_edge)


def set_seed(seed: int = 42):
    random.seed(seed)
    # Some libraries read this environment variable for deterministic hashing.
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy.
    np.random.seed(seed)
    # PyTorch CPU.
    torch.manual_seed(seed)
    # PyTorch GPU.
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
