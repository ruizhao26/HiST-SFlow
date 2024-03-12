import numpy as np
import os
import os.path as osp
import argparse
from encoding_utils import dat_to_spmat, SpikeToRaw, make_dir


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--data_root', type=str, default='/data/rzhao/scflow_dataset')
parser.add_argument('-d', '--dataset', type=str, default='spift')
parser.add_argument('-s', '--save_dir', type=str, default='dat_clip')
parser.add_argument('-l', '--data_length', type=int, default=25)
parser.add_argument('-sz', '--size', type=int, nargs='+', default=[500, 800])
parser.add_argument('-fud', '--flipud', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    data_path = osp.join(args.data_root, args.dataset)
    scene_path = sorted(os.listdir(data_path))
    for ss in scene_path:
        dat_file_path = osp.join(data_path, ss, 'test.dat')
        spmat = dat_to_spmat(dat_file_path, size=args.size, flipud=args.flipud)
        
        c, h, w = spmat.shape
        half_length = (args.data_length - 1) // 2
        
        # loop for dt=10 and dt=20
        for dt in [10, 20]:
            cur_scene_dt_path = osp.join(data_path, ss, 'dt{:d}'.format(dt))
            cur_save_path = osp.join(cur_scene_dt_path, args.save_dir+'_{:d}'.format(args.data_length))
            make_dir(cur_save_path)
            
            data_step = dt
            ii = 0
            while True:
                central_index = ii *data_step
                st_index = central_index - half_length
                ed_index = central_index + half_length + 1

                if (ed_index >= c - 40):
                    break

                if (central_index < 40):
                    ii += 1
                    continue

                cur_clip = spmat[st_index:ed_index, :, :]
                cur_clip_path = osp.join(cur_save_path, '{:04d}.dat'.format(ii))
                SpikeToRaw(save_path=cur_clip_path, SpikeSeq=cur_clip, flipud=args.flipud, delete_if_exists=True)
                print('finish ', cur_clip_path)

                ii += 1