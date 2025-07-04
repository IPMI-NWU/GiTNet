import os
import time
import random
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from medpy.metric.binary import dc
from Data import get_dataloader
from Method.models.ViGUNet import ViGUNet

def get_args_parser():
    parser = argparse.ArgumentParser('Full', add_help=False)
    parser.add_argument('--dataset', default='Kvasir', type=str) # Kvasir, NCI
    parser.add_argument('--output_dir', default='output/Kvasir/')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Full supervised training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format('0')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ViGUNet()
    model_path = args.output_dir + '0.8131.pth'

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(args.device)
    model.eval()
    test_loader = get_dataloader(args, split="test", resize_label=False)

    dice_score_list = []
    my_dice_score_list = []
    for data in test_loader:
        start = time.time()
        img, label = data['image'], data['label']
        img, label = img.to(args.device), label.to(args.device)

        _, output = model(img)
        output = output[-1]
        output = F.interpolate(output, size=label.shape[2:], mode="bilinear")
        output = nn.Sigmoid()(output)
        output = torch.where(output > 0.5, 1, 0)

        dice_score_list.append(dc(label, output))

        reduce_axis = list(range(2, len(img.shape)))
        intersection = torch.sum(output * label, dim=reduce_axis)
        input_o = torch.sum(output, dim=reduce_axis)
        target_o = torch.sum(label, dim=reduce_axis)
        my_dice = torch.mean(2 * intersection / (input_o + target_o + 1e-10), dim=1)
        my_dice_score_list.append(my_dice.item())

    dice_score = np.array(dice_score_list).mean()
    print(dice_score)

    dice_score = np.array(my_dice_score_list).mean()
    print(dice_score)
