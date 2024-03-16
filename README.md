## [AAAI 2024] Optical Flow for Spike Camera with Hierarchical Spatial-Temporal Spike Fusion

<h4 align="center"> Rui Zhao<sup>1</sup>, Ruiqin Xiong<sup>1</sup>, Jian Zhang<sup>2</sup>, Xinfeng Zhang<sup>3</sup>, Zhaofei Yu<sup>1,4</sup>, Tiejun Huang<sup>1,4</sup> </h4>
<h4 align="center">1. National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University<br>
2. School of Electronic and Computer Engineering, Peking University Shenzhen Graduate School<br>
3.  School of Computer Science and Technology, University of Chinese Academy of Sciences<br>
4.  Institute for Artificial Intelligence, Peking University
</h4><br>


This repository contains the official source code for our paper:

Optical Flow for Spike Camera with Hierarchical Spatial-Temporal Spike Fusion

AAAI 2024

## Environment

You can choose cudatoolkit version to match your server. The code is tested on PyTorch 2.0.1+cu120.

```bash
conda create -n hist python==3.10.9
conda activate hist
# You can choose the PyTorch version you like, we recommand version >= 1.10.1
# For example
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

## Prepare the Data

##### 1. Download and deploy the SPIFT and PHM dataset

[Link of SPIFT and PHM (BaiduNetDisk)](https://pan.baidu.com/s/1A5U9lsNyViGEQIyulSE8vg)  (password: 5331)

##### 2. Set the path of SPIFT and PHM dataset in your server

In lines 2-3 of `configs/flow.yml`

##### 3. Pre-processing for SPIFT and PHM

```shell
cd encoding &&
python encoding_dat.py --data_root 'your_root' --dataset spift --data_length 25 &&
python encoding_dat.py --data_root 'your_root' --dataset phm --data_length 25
```

## Evaluate

```shell
python main.py --arch hist_sflow --eval --pretrained ckpt/hist_sflow.pth
```

You can also inference optical flow from real data through `demo.py`

## Train

```shell
python3 main.py \ 
--learning_rate 1e-4 \ 
--configs ./configs/flow.yml \
--arch cr_rep25_conv1d_v2 \
--decay_factor 0.8 \
--vis_path ./vis/hist_sflow \
--save_name hist_sflow \
--eval_vis ./eval_vis/hist_sflow \
--weight_rec_loss 0.5 \
--scene_weight_list_type 1
```

We recommended to redirect the output logs by adding `>> hist_sflow.txt 2>&1` to the last of the above command for management.

## Citations

If you find this code useful in your research, please consider citing our paper. AAAI version:

```
@inproceedings{zhao2024optical,
  title={Optical Flow for Spike Camera with Hierarchical Spatial-Temporal Spike Fusion},
  author={Zhao, Rui and Xiong, Ruiqin and Zhang, Jian and Zhang, Xinfeng and Yu, Zhaofei and Huang, Tiejun},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}
```

## Acknowledgement

Parts of this code were derived from [askerlee/craft](https://github.com/askerlee/craft). Please also consider to cite [CRAFT](https://openaccess.thecvf.com/content/CVPR2022/html/Sui_CRAFT_Cross-Attentional_Flow_Transformer_for_Robust_Optical_Flow_CVPR_2022_paper.html) if you'd like to cite our paper.

