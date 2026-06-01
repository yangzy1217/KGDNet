import os
import math
import time
import argparse
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.autograd
from skimage import io
from torchvision.transforms import functional as transF
from torch.cuda.amp import autocast
from tqdm import tqdm

from models.KGDNet_simple import KGDNet as Net
from datasets import cd_dataset as Data

NET_NAME = 'KGDNet'

# Select one of: 'LEVIR-CD-256', 'LEVIR-CD+-256', 'SYSU-CD', 'WHU-CD'.
DATA_NAME = 'WHU-CD'
DATA_ROOT = Data.get_root(DATA_NAME)
ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
SAVE_DIR = f"{DATA_NAME}_{ts}"


class PredOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--crop_size', required=False, default=(1024, 1024), help='cropping size')
        parser.add_argument('--TTA', required=False, default=True, help='Test time augmentation')
        parser.add_argument('--test_dir', required=False, default=os.path.join(DATA_ROOT, 'test'),
                            help='directory to test images')
        parser.add_argument('--pred_dir', required=False,
                            default=os.path.join(working_path, 'eval', SAVE_DIR, NET_NAME),
                            help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default=os.path.join(
            working_path,
            f'/home/dgxuser/Students/YZY/CD/KGDNet/checkpoints/WHU-CD/20260601/WHU-CD_best_epoch0_OA0.9826_F10.8064_IoU0.6776.pth'
        ))
        parser.add_argument('--dev_id', required=False, default=0, help='Device id')
        parser.add_argument('--threshold', required=False, default=0.5, type=float, help='Prediction threshold')

        # Metric-related options.
        parser.add_argument('--gt_dir', required=False,
                            default=f'/home/dgxuser/Students/YZY/Dataset/{DATA_NAME}/test/label',
                            help='directory to ground-truth masks')
        parser.add_argument('--metrics_log', required=False, default=True, type=bool,
                            help='whether to save per-image metrics and summary to a .log file')
        parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation / inference')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


@torch.no_grad()
def inference_batch(net, batchA, batchB, batchA_d, batchB_d, opt):
    # batchA shape: [Batch_Size, C, H, W]
    with autocast():
        output = net(batchA, batchB, batchA_d, batchB_d)
        output = torch.sigmoid(output)

        if opt.TTA:
            # Vertical flip.
            out_v = net(torch.flip(batchA, [2]), torch.flip(batchB, [2]),
                        torch.flip(batchA_d, [2]), torch.flip(batchB_d, [2]))
            out_v = torch.sigmoid(torch.flip(out_v, [2]))

            # Horizontal flip.
            out_h = net(torch.flip(batchA, [3]), torch.flip(batchB, [3]),
                        torch.flip(batchA_d, [3]), torch.flip(batchB_d, [3]))
            out_h = torch.sigmoid(torch.flip(out_h, [3]))

            # Horizontal and vertical flip.
            out_hv = net(torch.flip(batchA, [2, 3]), torch.flip(batchB, [2, 3]),
                         torch.flip(batchA_d, [2, 3]), torch.flip(batchB_d, [2, 3]))
            out_hv = torch.sigmoid(torch.flip(out_hv, [2, 3]))

            output = (output + out_v + out_h + out_hv) / 4.0

    return (output.detach() > opt.threshold).float().cpu().numpy()


def create_crops(imgA, imgB, imgA_depth, imgB_depth, size):
    """Create sliding crops for RGB and depth image pairs."""
    imgA_crops, imgB_crops = [], []
    imgA_crops_depth, imgB_crops_depth = [], []
    h, w = imgA.shape[0], imgA.shape[1]
    c_h, c_w = size
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
        return 1
    rows = math.ceil(h / c_h)
    cols = math.ceil(w / c_w)
    stride_h = int((c_h * rows - h) / max(rows - 1, 1))
    stride_w = int((c_w * cols - w) / max(cols - 1, 1))
    for j in range(rows):
        for i in range(cols):
            s_h = int(j * c_h - j * stride_h)
            if j == rows - 1:
                s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i * c_w - i * stride_w)
            if i == cols - 1:
                s_w = w - c_w
            e_w = s_w + c_w
            imgA_crops.append(imgA[s_h:e_h, s_w:e_w, :])
            imgB_crops.append(imgB[s_h:e_h, s_w:e_w, :])
            imgA_crops_depth.append(imgA_depth[s_h:e_h, s_w:e_w, :])
            imgB_crops_depth.append(imgB_depth[s_h:e_h, s_w:e_w, :])
    print('Sliding crop finished. %d images created.' % len(imgA_crops))
    return imgA_crops, imgB_crops, imgA_crops_depth, imgB_crops_depth


def stitch_pred(patch_list, size_stitch):
    """Stitch cropped predictions back to the original image size."""
    H, W = size_stitch
    h, w = patch_list[0].shape
    stitch_rows = math.ceil(H / h)
    stitch_cols = math.ceil(W / w)
    assert stitch_rows * stitch_cols == len(patch_list), "Stitching patch number mismatch."

    h_overlap = int((h * stitch_rows - H) / max(stitch_rows - 1, 1))
    w_overlap = int((w * stitch_cols - W) / max(stitch_cols - 1, 1))

    rows_assembled = []
    idx = 0
    for r in range(stitch_rows):
        # Crop vertical overlap.
        crop_t = 0 if r == 0 else math.ceil(h_overlap / 2)
        crop_b = 0 if r == stitch_rows - 1 else (h_overlap - math.ceil(h_overlap / 2))

        first = patch_list[idx][crop_t:h - crop_b, :]
        idx += 1
        row = first

        for c in range(1, stitch_cols):
            # Crop horizontal overlap.
            crop_l = 0 if c == 0 else math.ceil(w_overlap / 2)
            crop_r = 0 if c == stitch_cols - 1 else (w_overlap - math.ceil(w_overlap / 2))
            piece = patch_list[idx][crop_t:h - crop_b, crop_l:w - crop_r]
            idx += 1
            row = np.concatenate((row, piece), axis=1)

        rows_assembled.append(row)

    stitched_img = rows_assembled[0]
    for r in range(1, len(rows_assembled)):
        stitched_img = np.concatenate((stitched_img, rows_assembled[r]), axis=0)

    stitched_img = stitched_img[:H, :W]
    print('Pred Stitched (%d, %d)' % (stitched_img.shape[0], stitched_img.shape[1]))
    return stitched_img


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def metrics_from_confusion(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    oa = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    miou = (iou_fg + iou_bg) / 2.0

    # Background-class precision, recall, and F1, treating 0 as positive.
    precision_bg = safe_div(tn, tn + fn)
    recall_bg = safe_div(tn, tn + fp)
    f1_bg = safe_div(2 * precision_bg * recall_bg, precision_bg + recall_bg)
    mf1 = (f1 + f1_bg) / 2.0

    return dict(
        OA=float(oa),
        Precision=float(precision),
        Recall=float(recall),
        F1=float(f1),
        IoU=float(iou_fg),
        mIoU=float(miou),
        mF1=float(mf1)
    )


def accumulate_confusion(pred_bin, gt_bin):
    # pred_bin and gt_bin are expected to be 0/1 arrays or boolean masks.
    pred = pred_bin.astype(np.uint8).reshape(-1)
    gt = gt_bin.astype(np.uint8).reshape(-1)
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, tn, fn


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net()
    print(opt.chkpt_path)
    state = torch.load(opt.chkpt_path, map_location="cpu")

    # Accept either {'model': state_dict} or a raw state_dict.
    state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state

    # Strip a leading DataParallel/DDP 'module.' prefix if present.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        nk = k.replace('module.', '', 1) if k.startswith('module.') else k
        new_state_dict[nk] = v

    # First load loosely so missing and unexpected keys can be reported.
    missing, unexpected = net.load_state_dict(new_state_dict, strict=False)
    if missing:
        print('[warn] Missing keys:', len(missing))
        for k in missing[:20]:
            print('   ', k)
        if len(missing) > 20:
            print('   ...', len(missing) - 20, 'more')
    if unexpected:
        print('[warn] Unexpected keys:', len(unexpected))
        for k in unexpected[:20]:
            print('   ', k)
        if len(unexpected) > 20:
            print('   ...', len(unexpected) - 20, 'more')

    # Strictly reload once to ensure the checkpoint fully matches.
    net.load_state_dict(new_state_dict, strict=True)
    net.to(torch.device('cuda', int(opt.dev_id))).eval()

    summary = predict_and_eval(net, opt)

    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)

    print('\n===== Dataset Summary =====')
    for k, v in summary.items():
        print(f'{k}: {v:.6f}')


def predict_and_eval(net, opt):
    if not os.path.exists(opt.pred_dir):
        os.makedirs(opt.pred_dir)
    if not os.path.exists(opt.gt_dir):
        raise FileNotFoundError(f'gt_dir not found: {opt.gt_dir}')

    # Batch size can be adjusted according to available GPU memory.
    BATCH_SIZE = getattr(opt, 'batch_size', 32)
    print(f"Accelerating Small Images with Multi-Image Batching (Size: {BATCH_SIZE})...")

    imgA_dir = os.path.join(opt.test_dir, 'A')
    imgB_dir = os.path.join(opt.test_dir, 'B')
    imgA_depth_dir = os.path.join(opt.test_dir, 'A_depth')
    imgB_depth_dir = os.path.join(opt.test_dir, 'B_depth')

    data_list = os.listdir(imgA_dir)
    valid_list = [it for it in data_list if os.path.splitext(it)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    print(f"Total images: {len(valid_list)}")

    device = torch.device('cuda', int(opt.dev_id))

    total_tp = total_fp = total_tn = total_fn = 0
    per_image_rows = []

    batch_buffer = {
        'names': [], 'A': [], 'B': [], 'A_d': [], 'B_d': [], 'dims': []
    }

    def process_batch(buffer):
        nonlocal total_tp, total_fp, total_tn, total_fn

        if len(buffer['names']) == 0:
            return

        # Stack cached samples into one batch.
        tA = torch.stack(buffer['A']).to(device).float()
        tB = torch.stack(buffer['B']).to(device).float()
        tA_d = torch.stack(buffer['A_d']).to(device).float()
        tB_d = torch.stack(buffer['B_d']).to(device).float()

        # Run batch inference.
        preds_numpy = inference_batch(net, tA, tB, tA_d, tB_d, opt)

        # Save each prediction and accumulate metrics.
        for i, name in enumerate(buffer['names']):
            base_name = name

            # Ensure each prediction is shaped as [H, W].
            pred = preds_numpy[i]
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]
            pred = pred.astype(np.uint8)

            io.imsave(os.path.join(opt.pred_dir, base_name + '.png'),
                      pred * 255, check_contrast=False)

            gt_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                p = os.path.join(opt.gt_dir, base_name + ext)
                if os.path.exists(p):
                    gt_path = p
                    break

            if gt_path:
                gt = io.imread(gt_path)
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt_bin = (gt > 127).astype(np.uint8)

                # Accumulate metrics only when prediction and label sizes match.
                if gt_bin.shape == pred.shape:
                    tp, fp, tn, fn = accumulate_confusion(pred, gt_bin)
                    total_tp += tp
                    total_fp += fp
                    total_tn += tn
                    total_fn += fn

                    m = metrics_from_confusion(tp, fp, tn, fn)
                    per_image_rows.append([
                        base_name, m['OA'], m['Precision'], m['Recall'],
                        m['F1'], m['IoU'], m['mIoU'], m['mF1']
                    ])
                else:
                    print(f"[warn] Shape mismatch for {base_name}: "
                          f"pred {pred.shape}, gt {gt_bin.shape}")
            else:
                print(f"[warn] GT not found for {base_name}")

    for it in tqdm(valid_list, desc="Batch Predicting", ncols=80):
        base_name, ext = os.path.splitext(it)

        pA = os.path.join(imgA_dir, it)
        pB = os.path.join(imgB_dir, base_name + '.png') if os.path.exists(os.path.join(imgB_dir, base_name + '.png')) else os.path.join(imgB_dir, it)
        pAd = os.path.join(imgA_depth_dir, base_name + '.png')
        pBd = os.path.join(imgB_depth_dir, base_name + '.png')

        # This batching path assumes all images in a batch share the same size.
        iA = transF.to_tensor(Data.normalize_image(io.imread(pA)))
        iB = transF.to_tensor(Data.normalize_image(io.imread(pB)))
        iAd = transF.to_tensor(Data.normalize_image(io.imread(pAd)))
        iBd = transF.to_tensor(Data.normalize_image(io.imread(pBd)))

        batch_buffer['names'].append(base_name)
        batch_buffer['A'].append(iA)
        batch_buffer['B'].append(iB)
        batch_buffer['A_d'].append(iAd)
        batch_buffer['B_d'].append(iBd)

        if len(batch_buffer['names']) >= BATCH_SIZE:
            process_batch(batch_buffer)
            for k in batch_buffer:
                batch_buffer[k] = []

    # Process the final partial batch.
    if len(batch_buffer['names']) > 0:
        process_batch(batch_buffer)

    summary = metrics_from_confusion(total_tp, total_fp, total_tn, total_fn)

    # Export one key-value metrics log.
    if getattr(opt, 'metrics_log', False):
        log_path = os.path.join(opt.pred_dir, f'{DATA_NAME}_{NET_NAME}_metrics.log')
        with open(log_path, 'w') as f:
            f.write(f'RUN_TS={datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'BATCH_SIZE={BATCH_SIZE}\n')
            f.write('---\n')
            for row in per_image_rows:
                name, oa, prec, rec, f1, iou, miou, mf1 = row
                f.write(f'[IMAGE]={name}\nOA={oa:.6f}\nF1={f1:.6f}\nIoU={iou:.6f}\n\n')
            f.write('[SUMMARY]\n')
            f.write(f'OA={summary["OA"]:.6f}\nF1={summary["F1"]:.6f}\nIoU={summary["IoU"]:.6f}\n')
        print(f'Metrics log saved to: {log_path}')

    return summary


if __name__ == '__main__':
    main()
