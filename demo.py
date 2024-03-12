import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch

from configs.yml_parser import YAMLParser
from models.get_model import get_model
from utils import *
from visulization_utils import *

from datasets.ds_utils import dat_to_spmat, writeFlow

parser = argparse.ArgumentParser()
##----------------------------- About Data -----------------------------
parser.add_argument('--configs', '-c', type=str, default='./configs/flow.yml')
parser.add_argument('--arch', '-a', type=str, default='hist_sflow')
##----------------------------- About Model -----------------------------
parser.add_argument('--pretrained', '-prt', type=str, default='ckpt/hist_sflow.pth')
##----------------------------- About Saving -----------------------------
parser.add_argument('--eval_root', '-evr', type=str, default='eval_real')
args = parser.parse_args()

cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

device = 'cuda'

def eval_real(model):
    #--------------
    model.eval()
    #--------------

    scene = 'car'
    data_path = 'real_data/{:s}.dat'.format(scene)
    spikes = dat_to_spmat(data_path, size=(250, 400), flipud=True)

    time_stamp1 = 100
    time_stamp2 = 120

    seq1 = torch.from_numpy(spikes[time_stamp1-12:time_stamp1+13,:,:]).unsqueeze(dim=0).to(device).float()
    seq2 = torch.from_numpy(spikes[time_stamp2-12:time_stamp2+13,:,:]).unsqueeze(dim=0).to(device).float()

    padder = InputPadder(dims=(250, 400))

    seq1, seq2 = padder.pad(seq1, seq2)

    with torch.no_grad():
        flow_and_flow_low, res_dict = model(seq1=seq1, seq2=seq2)
    flow = flow_and_flow_low[0]
    flow = padder.unpad(flow)

    flow_numpy = flow[0].permute([1,2,0]).cpu().numpy()
    flow_vis = flow_to_img(flow_numpy, convert_to_bgr=True)
    os.makedirs('eval_real', exist_ok=True)
    vis_path = osp.join('eval_real', '{:s}_idx_{:d}_to_{:d}.png'.format(scene, time_stamp1, time_stamp2))
    flo_path = osp.join('eval_real', '{:s}_idx_{:d}_to_{:d}.flo'.format(scene, time_stamp1, time_stamp2))

    cv2.imwrite(vis_path, flow_vis)
    writeFlow(flo_path, flow_numpy)

    return


if __name__ == '__main__':
    model = get_model(args)

    if args.pretrained:
        load_data = torch.load(args.pretrained)
        if 'optimizer' in load_data.keys():     # new save model
            network_data = load_data['model']
            optimizer_data = load_data['optimizer'] 
        else:                                   # old save model
            network_data = load_data

        print('=> using pretrained flow model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(network_data)

    
    with torch.no_grad():
        eval_real(model=model)
