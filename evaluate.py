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
def visualize(args):
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
            pred_masks_raw = (pred_masks_raw*255).round().astype(np.uint8)
            cv2.imshow('Input Image', img_np)
            cv2.imshow('Ground truth', gt_mask)
            cv2.imshow('Pyramid attention network', pred_masks_raw)
            calculate_auc(pred_masks_raw, gt_mask, plot=True, model_name='Pyramid attention network')

            #CV2 saliency
            saliency_spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMapSpectral) = saliency_spectral.computeSaliency(img_np)
            saliencyMapSpectral = (saliencyMapSpectral * 255).round().astype(np.uint8)
            cv2.imshow("Static (spectral residual)", saliencyMapSpectral)
            calculate_auc(saliencyMapSpectral, gt_mask, plot=True, model_name='Static (spectral residual)')

            saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMapFG) = saliency_fg.computeSaliency(img_np)
            saliencyMapFG = (saliencyMapFG * 255).round().astype(np.uint8)
            cv2.imshow("Static (fine-grained)", saliencyMapFG)
            calculate_auc(saliencyMapFG, gt_mask, plot=True, model_name='Static (fine-grained)')

            key = cv2.waitKey(0)
            if key == ord('q'):
                break

def compare_methods(args):
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
    auc_pyramid = 0
    auc_spectral = 0
    auc_fg = 0
    count = 0
    with torch.no_grad():
        for _, (img_np, img_tor, gt_mask) in enumerate(eval_dataloader, start=1):
            gt_mask = np.squeeze(gt_mask.cpu().numpy(), axis=0)
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_raw = (pred_masks_raw*255).round().astype(np.uint8)
            auc_pyramid += calculate_auc(pred_masks_raw, gt_mask)

            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            saliency_spectral = cv2.saliency.StaticSaliencySpectralResidual_create()
            (success, saliencyMapSpectral) = saliency_spectral.computeSaliency(img_np)
            saliencyMapSpectral = (saliencyMapSpectral * 255).round().astype(np.uint8)
            auc_spectral += calculate_auc(saliencyMapSpectral, gt_mask)

            saliency_fg = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMapFG) = saliency_fg.computeSaliency(img_np)
            saliencyMapFG = (saliencyMapFG * 255).round().astype(np.uint8)
            auc_fg += calculate_auc(saliencyMapFG, gt_mask)
            count+=1
    print('Pyramid attention network: Average area under ROC curve: %f' % (auc_pyramid/count))
    print('CV2 static saliency (spectral): Average area under ROC curve: %f' % (auc_spectral/count))
    print('CV2 static saliency (fine-grained): Average area under ROC curve: %f' % (auc_fg/count))
    return auc_pyramid/count, auc_spectral/count, auc_fg/count

def calculate_auc(map, gt, plot=False, model_name=''):
    pos = gt.nonzero()
    neg = np.where(gt == 0)
    tpr = []
    fpr = []
    for thresh in np.unique(map):
        tp = (map[pos] >= thresh).sum()
        fp = (map[neg] >= thresh).sum()
        tpr.append(tp/len(pos[0]))
        fpr.append(fp/len(neg[0]))
    if (plot):
        plt.plot(fpr, tpr)
        plt.title(model_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
    return metrics.auc(fpr, tpr)


if __name__ == '__main__':
    rt_args = parse_arguments()
    #visualize(rt_args)
    compare_methods(rt_args)
