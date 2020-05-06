from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics
from model import SODModel
from dataloader import InfDataloader, SODLoader, EvalDataLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='./data/DUTS/DUTS-TE/DUTS-TE-Image', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='./models/alph-0.7_wbce_w0-1.0_w1-1.15/' + os.listdir('./models/alph-0.7_wbce_w0-1.0_w1-1.15/')[-1], help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

    return parser.parse_args()

#Visualize orginal/prediction/ground truth side-by-side
def compare(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    eval_data = EvalDataLoader(img_folder=args.imgs_folder, gt_path='./data/DUTS/DUTS-TE/DUTS-TE-Mask', target_size=args.img_size)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=True, num_workers=2)
    with torch.no_grad():
        for batch_idx, (img_np, img_tor, gt_mask) in enumerate(eval_dataloader, start=1):
            gt_mask = np.squeeze(gt_mask.cpu().numpy(), axis=0)
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)

            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_raw = (pred_masks_raw*255).astype(np.uint8)
            cv2.imshow('Input Image', img_np)
            cv2.imshow('Generated Saliency Mask', pred_masks_raw)
            cv2.imshow('Ground truth', gt_mask)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break

def calculate_auc(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    eval_data = EvalDataLoader(img_folder=args.imgs_folder, gt_path='./data/DUTS/DUTS-TE/DUTS-TE-Mask', target_size=args.img_size)
    eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=True, num_workers=2)
    auc = 0
    count = 0
    with torch.no_grad():
        for _, (_, img_tor, gt_mask) in enumerate(eval_dataloader, start=1):
            count+=1
            gt_mask = np.squeeze(gt_mask.cpu().numpy(), axis=0)
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_raw = (pred_masks_raw*255).astype(np.uint8)
            pos = gt_mask.nonzero()
            neg = np.where(gt_mask == 0)
            tpr = []
            fpr = []
            for thresh in range(256):
                tp = (pred_masks_raw[pos] >= thresh).sum()
                fp = (pred_masks_raw[neg] >= thresh).sum()
                tpr.append(tp/len(pos[0]))
                fpr.append(fp/len(neg[0]))
            auc += metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.show()
    print('Average area under ROC curve: %f' % (auc/count))
    return auc/count

if __name__ == '__main__':
    rt_args = parse_arguments()
    #compare(rt_args)
    calculate_auc(rt_args)
