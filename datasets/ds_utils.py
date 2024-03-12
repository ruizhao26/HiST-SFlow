import os
import os.path as osp
import numpy as np
import cv2


'''
# --------------------------------------------
# check if file exists
# --------------------------------------------
'''

def check_exist_list(p_list):
    for p in p_list:
        if not osp.exists(p):
            print(p, ' does not exists')
            return False
    
    return True


'''
# --------------------------------------------
# for reading spikes from dat files
# --------------------------------------------
'''

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
    sp_mat = RawToSpike(video_seq, size[0], size[1], flipud=flipud)
    
    return sp_mat


'''
# --------------------------------------------
# for reading spikes from dat files
# --------------------------------------------
'''

def read_img(img_path):
    '''
    read images -> (1 x H x W), float, [0, 1]
    '''
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32) / 255
    img = img[None, :, :]
    return img


'''
# --------------------------------------------
# for reading flow from .flo files
# --------------------------------------------
'''

TAG_CHAR = np.array([202021.25], np.float32)
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
