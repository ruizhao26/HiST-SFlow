import os
import os.path as osp
import numpy as np


def make_dir(p):
    if not osp.exists(p):
        os.makedirs(p)
    return


def RawToSpike(video_seq, h, w, flipud=False):
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)

    return SpikeMatrix


def dat_to_spmat(dat_path, size=(500, 800), flipud=False):
    f = open(dat_path, 'rb')
    video_seq = f.read()
    video_seq = np.frombuffer(video_seq, 'b')
    sp_mat = RawToSpike(video_seq, 500, 800, flipud=flipud)
    
    return sp_mat


# saving .dat file from spikes
def SpikeToRaw(save_path, SpikeSeq, flipud=True, delete_if_exists=True):
    """
        save spike sequence to .dat file
        save_path: full saving path (string)
        SpikeSeq: Numpy array (T x H x W)
        Rui Zhao
    """
    if delete_if_exists:
        if osp.exists(save_path):
            os.remove(save_path)

    sfn, h, w = SpikeSeq.shape
    assert (h * w) % 8 == 0
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        if flipud:
            # 模拟相机的倒像
            spike = np.flipud(SpikeSeq[img_id, :, :])
        else:
            spike = SpikeSeq[img_id, :, :]
        # numpy按自动按行排，数据也是按行存的
        spike = spike.flatten()
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())

    fid.close()

    return