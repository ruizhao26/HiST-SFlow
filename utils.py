import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import numpy as np
import cv2
import os
import os.path as osp


def make_dir(p):
    if not osp.exists(p):
        os.makedirs(p)
    return


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

class InputPadder:
    """ Pads images such that dimensions are divisible by pad_size """
    def __init__(self, dims, pad_size=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // pad_size) + 1) * pad_size - self.ht) % pad_size
        pad_wd = (((self.wd // pad_size) + 1) * pad_size - self.wd) % pad_size
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def supervised_loss(flow, flow_gt, gamma=0.8):
    n_predictions = len(flow)
    flow_loss = 0.0
    # i代表raft的decoder的第i轮iteration输出的光流，从flow_predictions这个list里解析处一个bchw
    # j代表从bchw里取挨个的b
    # 这个loss写麻烦了，和所有的b并行做loss等价
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        for j in range(len(flow_gt)):
            i_loss = (flow[i][j] - flow_gt[j]).abs()
            flow_loss += i_weight * i_loss.mean() / len(flow_gt)
    return flow_loss


def supervised_loss_scflow(flow_preds, flow_gt):
    w_scales = [1,1,1,1]
    res_dict = {}
    pym_losses = []
    _, _, H, W = flow_gt.shape
    
    for i, flow in enumerate(flow_preds):
        b, c, h, w = flow.shape
        flowgt_scaled = F.interpolate(flow_gt, (h, w), mode='bilinear') * (h / H)

        curr_loss = (flowgt_scaled - flow).abs().mean()
        pym_losses.append(curr_loss)
    
    loss = [l * int(w) for l, w in zip(pym_losses, w_scales)]
    loss = sum(loss)
    return loss

def calculate_aepe(pred_flow, gt_flow):
    return torch.norm(pred_flow - gt_flow, p=2, dim=1).mean()

def calculate_error_rate(pred_flow, gt_flow):
    pred_flow = pred_flow.squeeze(dim=0).permute([1,2,0]).cpu().numpy()
    gt_flow = gt_flow.squeeze(dim=0).permute([1,2,0]).cpu().numpy()

    epe_map = np.sqrt(np.sum(np.square(pred_flow - gt_flow), axis=2))
    bad_pixels = np.logical_and(
        epe_map > 0.5,
        epe_map / np.maximum(
            np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
    return bad_pixels.mean() * 100.