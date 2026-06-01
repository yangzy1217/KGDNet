#!/usr/bin/env python3
import os
import argparse
import datetime
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.utils import binary_accuracy as accuracy, AverageMeter
from utils.utils import set_seed
from datasets.cd_dataset import RS, DATASET_CONFIGS
from models.KGDNet import KGDNet as Net

SUPPORTED_DATASETS = list(DATASET_CONFIGS)


# Argument parsing.
def parse_args():
    parser = argparse.ArgumentParser(description='Change Detection Training')
    parser.add_argument('--dataset-name', type=str, default='LEVIR-CD-256',
                        choices=SUPPORTED_DATASETS,
                        help='Dataset name, one of: ' + ', '.join(SUPPORTED_DATASETS))
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-tb', '--train-batch-size', type=int, default=16)
    parser.add_argument('-vb', '--val-batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--crop-nums', type=int, default=5)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--predict-step', type=int, default=1)
    parser.add_argument('--gpus', type=str, default=None,
                        help='comma-separated GPU device ids, or None for all')
    parser.add_argument('--multi-gpu', action='store_true',
                        help='use DataParallel across gpus')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--chkpt-dir', type=str, default='./checkpoints/')
    parser.add_argument('--pred-dir', type=str, default='./results')
    parser.add_argument('--freeze-sam', action='store_true',
                        help='Freeze SAM backbone weights')
    parser.add_argument('--test', action='store_true',
                        help='Run evaluation on test set after training')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs before using ReduceLROnPlateau')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='Optimizer to use (default: Adam)')
    return parser.parse_args()


def get_state_dict_safe(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


class _Tee(object):
    def __init__(self, stream, logfile):
        self.stream = stream
        self.log = logfile
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def _setup_run_logger(log_dir):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"train_{ts}.txt")
    logfile = open(log_path, 'w', encoding='utf-8', newline='')
    sys._orig_stdout = sys.stdout
    sys._orig_stderr = sys.stderr
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)
    return log_path, logfile


def _teardown_run_logger():
    logfile = getattr(sys.stdout, "log", None)
    try:
        sys.stdout = getattr(sys, "_orig_stdout", sys.stdout)
        sys.stderr = getattr(sys, "_orig_stderr", sys.stderr)
    finally:
        if logfile and not logfile.closed:
            logfile.close()


# Main training routine.
def main():
    args = parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    date_str = datetime.datetime.now().strftime('%Y%m%d')
    dataset_chkpt_dir = os.path.join(args.chkpt_dir, args.dataset_name, date_str)
    os.makedirs(dataset_chkpt_dir, exist_ok=True)
    log_path, _logfile = _setup_run_logger(dataset_chkpt_dir)
    print(f"[Logger] Saving console output to: {log_path}")
    dataset_log_dir = os.path.join(args.log_dir, args.dataset_name)
    dataset_pred_dir = os.path.join(args.pred_dir, args.dataset_name)
    os.makedirs(dataset_log_dir, exist_ok=True)
    os.makedirs(dataset_pred_dir, exist_ok=True)

    train_set = RS(dataset=args.dataset_name, mode='train',
                   random_crop=True, crop_nums=args.crop_nums,
                   crop_size=args.crop_size,
                   random_flip=True, random_temporal_flip=False)
    val_set = RS(dataset=args.dataset_name, mode='test',
                 random_crop=True, crop_nums=args.crop_nums,
                 crop_size=args.crop_size,
                 random_flip=True, random_temporal_flip=False)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            shuffle=False, num_workers=args.workers)

    net = Net().to(device)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net).to(device)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

    if args.optimizer == 'Adam':
        optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'SGD':
        optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=False)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    writer = SummaryWriter(dataset_log_dir)

    bestF = 0.0
    best_ckpt_path = None  # Track the current best checkpoint path.

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_epoch(train_loader, net, optimizer, writer, epoch, device, args)
        valF, val_acc, val_iou, val_loss = validate(val_loader, net, writer, epoch, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('lr', current_lr, epoch)
        print(f"LR: {current_lr:.8f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} F1: {valF:.4f} IoU: {val_iou:.4f}")

        # Save only the best model according to validation F1.
        if valF > bestF:
            bestF = valF

            # Remove the previous best checkpoint before saving the new one.
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                try:
                    os.remove(best_ckpt_path)
                    print(f"Removed previous best model: {best_ckpt_path}")
                except OSError as e:
                    print(f"Error removing previous best checkpoint: {e}")

            # Prepare and save the new best checkpoint.
            state_dict = get_state_dict_safe(net)
            ckpt_filename = (f'{args.dataset_name}_best_epoch{epoch}_'
                             f'OA{val_acc:.4f}_F1{bestF:.4f}_IoU{val_iou:.4f}.pth')
            best_ckpt_path = os.path.join(dataset_chkpt_dir, ckpt_filename)
            torch.save(state_dict, best_ckpt_path)
            print(f"Saved new best model: {best_ckpt_path}")

    if args.test:
        print("\n========== Running Test Set Evaluation ==========")
        # Fallback: if no best checkpoint was saved, load the latest .pth file.
        if best_ckpt_path is None or not os.path.exists(best_ckpt_path):
            pth_files = [f for f in os.listdir(dataset_chkpt_dir) if f.endswith('.pth')]
            if not pth_files:
                raise FileNotFoundError(f"No .pth files found in {dataset_chkpt_dir}")
            pth_files = sorted(pth_files, key=lambda f: os.path.getmtime(os.path.join(dataset_chkpt_dir, f)))
            best_ckpt_path = os.path.join(dataset_chkpt_dir, pth_files[-1])

        print(f"Loading model from: {best_ckpt_path}")
        net.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        testF, test_acc, test_iou, test_loss = validate(val_loader, net, writer=None, epoch=0, device=device)
        print(f"Test Loss: {test_loss:.4f} | Test F1: {testF:.4f} | Test IoU: {test_iou:.4f} | Test Acc: {test_acc:.2f}%")
        print("========== Test Set Evaluation Finished ==========")

    writer.close()
    _teardown_run_logger()


def train_epoch(loader, net, optimizer, writer, epoch, device, args):
    net.train()
    meter_loss = AverageMeter()
    meter_acc = AverageMeter()
    loop = tqdm(loader, desc=f"Train Epoch {epoch}")
    for i, (imgA, imgB, A_depth, B_depth, labels) in enumerate(loop, 1):
        imgA, imgB = imgA.to(device), imgB.to(device)
        A_depth, B_depth = A_depth.to(device), B_depth.to(device)
        labels = labels.to(device).unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = net(imgA, imgB, A_depth, B_depth)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        meter_loss.update(loss.item())
        preds = torch.sigmoid(outputs)
        acc = ((preds > 0.5) == labels).float().mean().item()
        meter_acc.update(acc)

        loop.set_postfix(loss=f"{meter_loss.val:.4f}", acc=f"{meter_acc.val * 100:.2f}%")
        if i % args.print_freq == 0:
            step = epoch * len(loader) + i
            writer.add_scalar('train/loss', meter_loss.val, step)
            writer.add_scalar('train/acc', meter_acc.val, step)

    return meter_loss.avg, meter_acc.avg


def validate(loader, net, writer, epoch, device):
    net.eval()
    meter_F1 = AverageMeter()
    meter_acc = AverageMeter()
    meter_IoU = AverageMeter()
    meter_loss = AverageMeter()
    loop = tqdm(loader, desc=f"Val Epoch {epoch}")
    with torch.no_grad():
        for i, (imgA, imgB, A_depth, B_depth, labels) in enumerate(loop, 1):
            imgA, imgB = imgA.to(device), imgB.to(device)
            A_depth, B_depth = A_depth.to(device), B_depth.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = net(imgA, imgB, A_depth, B_depth)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            meter_loss.update(loss.item())

            preds = torch.sigmoid(outputs)
            acc, _, _, F1, IoU = accuracy(preds.squeeze().cpu().numpy(), labels.squeeze().cpu().numpy())
            meter_acc.update(acc)
            meter_F1.update(F1)
            meter_IoU.update(IoU)

            loop.set_postfix(loss=f"{meter_loss.val:.4f}",
                             F1=f"{meter_F1.val * 100:.2f}%",
                             IoU=f"{meter_IoU.val * 100:.2f}%")

        if writer is not None:
            writer.add_scalar('val/loss', meter_loss.avg, epoch)
            writer.add_scalar('val/F1', meter_F1.avg, epoch)
            writer.add_scalar('val/IoU', meter_IoU.avg, epoch)
            writer.add_scalar('val/acc', meter_acc.avg, epoch)

    return meter_F1.avg, meter_acc.avg, meter_IoU.avg, meter_loss.avg


if __name__ == '__main__':
    set_seed(1217)
    main()
