import os
import os.path as osp
import torch
import numpy as np
from datasets.ds_utils import *


class Augmentor:
    def __init__(self, crop_size, do_flip):
        # spatial augmentation params
        self.crop_size = crop_size
        self.spatial_aug_prob = 0.8

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.2

    def spatial_transform(self, spk_list, flow):
        y0 = np.random.randint(0, spk_list[0].shape[1] - self.crop_size[0])
        x0 = np.random.randint(0, spk_list[0].shape[2] - self.crop_size[1])
        do_lr_flip = np.random.rand() < self.h_flip_prob
        do_ud_flip = np.random.rand() < self.v_flip_prob

        for ii, spk in enumerate(spk_list):
            if self.do_flip:
                if do_lr_flip:
                    spk = np.flip(spk, axis=2)
                if do_ud_flip:
                    spk = np.flip(spk, axis=1)
            spk = spk[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            spk_list[ii] = spk
        
        if self.do_flip:
            if do_lr_flip:
                flow = np.flip(flow, axis=2)
                flow[0,:,:] = -flow[0,:,:]
            if do_ud_flip:
                flow = np.flip(flow, axis=1)
                flow[1,:,:] = -flow[1,:,:]
        flow = flow[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return spk_list, flow

    def __call__(self, seq1, seq2, img1, img2, flow):
        spk_list = [seq1, seq2, img1, img2]
        spk_list, flow = self.spatial_transform(spk_list, flow)
        spk_list = [np.ascontiguousarray(spk) for spk in spk_list]
        seq1, seq2, img1, img2 = spk_list[0], spk_list[1], spk_list[2], spk_list[3]
        flow = np.ascontiguousarray(flow)
        return seq1, seq2, img1, img2, flow


class SpiftDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_root = self.cfg['data']['spift_path']
        self.augmentor = Augmentor(crop_size=self.cfg['loader']['crop_size'], do_flip=self.cfg['loader']['do_flip'])
        self.samples = self.collect_samples()
        print('Training data have {:d} samples'.format(len(self.samples)))

    def collect_samples(self):
        scene_list = sorted(os.listdir(self.data_root))
        samples = []
        for scene in scene_list:
            # dt10与dt20混合
            for dt in [10, 20]:
                cur_scene_dt_path = osp.join(self.data_root, scene, 'dt{:d}'.format(dt))
                dat_dir = osp.join(cur_scene_dt_path, 'dat_clip_25')
                img_dir = osp.join(cur_scene_dt_path, 'imgs')
                flowgt_dir = osp.join(cur_scene_dt_path, 'flow')
                
                dat_list = sorted(os.listdir(dat_dir))
                for dat_idx in range(len(dat_list) - 1):
                    seq1_path = osp.join(dat_dir, dat_list[dat_idx])
                    seq2_path = osp.join(dat_dir, dat_list[dat_idx+1])

                    img1_path = osp.join(img_dir, dat_list[dat_idx][:-4]+'.png')
                    img2_path = osp.join(img_dir, dat_list[dat_idx+1][:-4]+'.png')

                    seq1_name = dat_list[dat_idx][:-4] + '.flo'
                    flow_path = osp.join(flowgt_dir, seq1_name)

                    if check_exist_list([seq1_path, seq2_path, flow_path]):
                        s = {}
                        s['seq1_path'] = seq1_path
                        s['seq2_path'] = seq2_path
                        s['img1_path'] = img1_path
                        s['img2_path'] = img2_path
                        s['flow_path'] = flow_path
                        samples.append(s)
        return samples
    
    def _load_sample(self, s):
        # init shape when reading
        # seq1 and seq2: T x H x W
        # flow: H x W x 2
        data = {}
        data['seq1'] = dat_to_spmat(dat_path=s['seq1_path'], size=(500,800), flipud=False).astype(np.float32)
        data['seq2'] = dat_to_spmat(dat_path=s['seq2_path'], size=(500,800), flipud=False).astype(np.float32)
        data['img1'] = read_img(img_path=s['img1_path'])
        data['img2'] = read_img(img_path=s['img2_path'])
        flow = readFlow(s['flow_path']).astype(np.float32)
        data['flow'] = np.transpose(flow, (2,0,1))

        data['seq1'], data['seq2'], data['img1'], data['img2'], data['flow'] = self.augmentor(data['seq1'], data['seq2'], data['img1'], data['img2'], data['flow'])
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data


def get_phm_test_set(cfg, scene_list, dt_list, log):
    assert scene_list != None and scene_list != []
    assert dt_list != None and dt_list != []
    dt10_ds_list = []
    dt20_ds_list = []

    for ss in scene_list:
        if 10 in dt_list:
            dt10_ds_list.append(PhmDataset(cfg=cfg, scene=ss, dt=10))
        if 20 in dt_list:
            dt20_ds_list.append(PhmDataset(cfg=cfg, scene=ss, dt=20))

    log.info('information about test set:')
    for ii, ss in enumerate(scene_list):
        out_str = 'scene : {:6s}'.format(ss)
        if 10 in dt_list:
            out_str += 'dt10: {:4d} samples     '.format(len(dt10_ds_list[ii]))
        if 20 in dt_list:
            out_str += 'dt20: {:4d} samples'.format(len(dt20_ds_list[ii]))
        log.info(out_str)

    return dt10_ds_list, dt20_ds_list


class PhmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, scene, dt):
        self.cfg = cfg
        self.data_root = self.cfg['data']['phm_path']
        self.scene = scene
        self.dt = dt
        self.samples = self.collect_samples()
        # print('Training data have {:d} samples'.format(len(self.samples)))

    def collect_samples(self):
        scene_list = sorted(os.listdir(self.data_root))
        samples = []
        cur_scene_dt_path = osp.join(self.data_root, self.scene, 'dt{:d}'.format(self.dt))
        dat_dir = osp.join(cur_scene_dt_path, 'dat_clip_25')
        flowgt_dir = osp.join(cur_scene_dt_path, 'flow')
        
        dat_list = sorted(os.listdir(dat_dir))
        for dat_idx in range(len(dat_list) - 1):
            seq1_path = osp.join(dat_dir, dat_list[dat_idx])
            seq2_path = osp.join(dat_dir, dat_list[dat_idx+1])

            seq1_name = dat_list[dat_idx][:-4] + '.flo'
            flow_path = osp.join(flowgt_dir, seq1_name)

            if check_exist_list([seq1_path, seq2_path, flow_path]):
                s = {}
                s['seq1_path'] = seq1_path
                s['seq2_path'] = seq2_path
                s['flow_path'] = flow_path
                samples.append(s)
        return samples
    
    def _load_sample(self, s):
        # init shape when reading
        # seq1 and seq2: T x H x W
        # flow: H x W x 2
        data = {}
        data['seq1'] = dat_to_spmat(dat_path=s['seq1_path'], size=(500,800), flipud=False).astype(np.float32)
        data['seq2'] = dat_to_spmat(dat_path=s['seq2_path'], size=(500,800), flipud=False).astype(np.float32)
        flow = readFlow(s['flow_path']).astype(np.float32)
        data['flow'] = np.transpose(flow, (2,0,1))

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = self._load_sample(self.samples[index])
        return data