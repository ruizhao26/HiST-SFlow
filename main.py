import argparse
import os
import os.path as osp
import time
import datetime
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pprint

from configs.yml_parser import YAMLParser
from datasets.datasets import *
from models.get_model import get_model
from logger import init_logger
from utils import *
from visulization_utils import *

parser = argparse.ArgumentParser()
##----------------------------- About Data -----------------------------
parser.add_argument('--configs', '-c', type=str, default='./configs/flow.yml')
parser.add_argument('--arch', '-a', type=str, default='hist_sflow')
parser.add_argument('--save_dir', '-sd', type=str, default='./outputs')
parser.add_argument('--save_name', '-sn', type=str, default=None)
parser.add_argument('--eval', '-e', action='store_true')
##----------------------------- About Model -----------------------------
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--start-epoch', '-se', type=int, default=0)
parser.add_argument('--epochs', type=int, default=None)
##----------------------------- About Training HyperParams -----------------------------
parser.add_argument('--batch_size', '-bs', type=int, default=6)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--decay_freq', '-dfreq', type=int, default=10)
parser.add_argument('--decay_factor', '-dfac', type=float, default=0.7)
##----------------------------- About Training NOT Related to Results -----------------------------
parser.add_argument('--num_workers', '-j', type=int, default=12)
parser.add_argument('--print_freq', '-pf', type=int, default=100)
parser.add_argument('--valid_freq', '-valif', type=int, default=20)
parser.add_argument('--save_optimizer', type=int, default=True)
##----------------------------- About Training Visualization -----------------------------
parser.add_argument('--vis_freq', '-vf', type=int, default=40)
parser.add_argument('--vis_path', '-vp', type=str, default='./vis')
parser.add_argument('--eval_vis', '-ev', type=str, default='eval_vis')
parser.add_argument('--eval_vis_freq', '-evf', type=int, default=10)
##----------------------------- About rec Loss -----------------------------
parser.add_argument('--weight_rec_loss', '-wrec', type=float, default=0.2)

parser.add_argument('--scene_weight_list_type', type=int, default=1)

args = parser.parse_args()

n_iter = 0

cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config
cfg['train']['decay_freq'] = args.decay_freq
cfg['train']['decay_factor'] = args.decay_factor

if args.epochs != None:
    cfg['loader']['n_epochs'] = args.epochs

if args.scene_weight_list_type == 1:
    args.scene_weight_list = [1, 0.5, 0.25, 0.125]

print('using scene_weight_list', args.scene_weight_list)


def cal_feat_loss(img_gt, rec_img_list, weight_list=[1, 0.5, 0.25, 0.125]):
    loss = 0.0
    for ii, rec_img in enumerate(rec_img_list):
        if ii != 0:
            img_gt_resize = torch.nn.functional.interpolate(img_gt, rec_img.shape[-2:], mode='bilinear')
        else:
            img_gt_resize = img_gt
        cur_loss = weight_list[ii] * (rec_img - img_gt_resize).abs().mean()
        loss = loss + cur_loss
    return loss


def train(cfg, train_loader, model, optimizer, epoch, scheduler, log, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sup_losses = AverageMeter()
    feat_losses = AverageMeter()
    model.train()
    end = time.time()

    ######################################################################
    ## Training Loop
    ## 监督学习

    for ww, data in enumerate(train_loader, 0):

        st1 = time.time()
        seq1 = data['seq1'].cuda()
        seq2 = data['seq2'].cuda()
        img1 = data['img1'].cuda()
        img2 = data['img2'].cuda()
        flowgt = data['flow'].cuda()
        data_time.update(time.time() - end)

        ##--------------- Forward -----------------
        flow, feat_list = model(seq1=seq1, seq2=seq2)
        ##-----------------------------------------

        ## compute loss
        sup_loss = supervised_loss(flow, flowgt)
        

        feat1_loss = cal_feat_loss(img_gt=img1, rec_img_list=feat_list[0], weight_list=args.scene_weight_list)
        feat2_loss = cal_feat_loss(img_gt=img2, rec_img_list=feat_list[1], weight_list=args.scene_weight_list)
        feat_loss = args.weight_rec_loss * (feat1_loss + feat2_loss)
        

        loss = sup_loss + feat_loss

        ## record loss
        losses.update(loss.item())
        sup_losses.update(sup_loss.item())
        feat_losses.update(feat_loss.item())
        
        flow_mean = flow[-1].abs().mean()
        if ww % 10 == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)
            train_writer.add_scalar('sup_loss', sup_loss.item(), n_iter)
            train_writer.add_scalar('feat_loss', feat_loss.item(), n_iter)
            train_writer.add_scalar('flow_mean', flow_mean, n_iter)

        ## compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # record elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        n_iter += 1
        if n_iter % args.vis_freq == 0:
            vis_flow_batch(flow[-1], args.vis_path, suffix='forw_flow', max_batch=16)

        ## output logs
        if ww % args.print_freq == 0:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            out_str = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
            out_str += 'Time: {},  Data: {},  Loss: {}, Sup Loss: {}, Feat Loss: {}, Flow mean {:.4f}, lr {:.7f}'.format(batch_time, data_time, losses, sup_losses, feat_losses, flow_mean, cur_lr)
            log.info(out_str)

        end = time.time()
    
    return


def validate(cfg, test_loader_lists, scene_list, dt_list, model, log):
    global n_iter

    #--------------
    model.eval()
    #--------------
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model_time = AverageMeter()

    test_loader_list_dt10 = test_loader_lists[0]
    test_loader_list_dt20 = test_loader_lists[1]
    dt10_metrics_avg = {'aepe': [], 'f1': []}
    dt20_metrics_avg = {'aepe': [], 'f1': []}

    padder = InputPadder(dims=(500, 800))

    if 10 in dt_list:
        assert len(test_loader_list_dt10) == len(scene_list)
    if 20 in dt_list:
        assert len(test_loader_list_dt20) == len(scene_list)

    for ii_scene, cur_scene in enumerate(scene_list):
        for cur_dt in dt_list:
            AEE = AverageMeter()
            F1 = AverageMeter()
            scene_eval_vis_path = osp.join(args.eval_vis, cur_scene, 'dt{:d}'.format(cur_dt))
            make_dir(scene_eval_vis_path)

            scene_loader_cur_dt = eval('test_loader_list_dt{:d}[ii_scene]'.format(cur_dt))
            for ww, data in enumerate(scene_loader_cur_dt, 0):
                st1 = time.time()
                seq1 = data['seq1'].cuda()
                seq2 = data['seq2'].cuda()
                flowgt = data['flow'].cuda()
                data_time.update(time.time() - st1)

                seq1, seq2 = padder.pad(seq1, seq2)

                with torch.no_grad():
                    st = time.time()
                    preds, feat_list = model(seq1=seq1, seq2=seq2)
                    flow, flow_low = preds
                    mtime = time.time() - st
                
                # flow = [padder.unpad(flow_elem) for flow_elem in flow]
                flow = padder.unpad(flow)

                # Visualization
                if ww % args.eval_vis_freq == 0:
                    flow_vis = flow_to_img(flow[0].permute([1,2,0]).cpu().numpy(), convert_to_bgr=True)
                    cur_vis_path = osp.join(scene_eval_vis_path, '{:04d}.png'.format(ww))
                    cv2.imwrite(cur_vis_path, flow_vis)

                # update metrics
                aepe = calculate_aepe(flow[0], flowgt)
                f1 = calculate_error_rate(flow[0], flowgt)

                model_time.update(mtime)
                AEE.update(aepe)
                F1.update(f1)

            if cur_dt == 10:
                dt10_metrics_avg['aepe'].append(AEE.avg)
                dt10_metrics_avg['f1'].append(F1.avg)
            elif cur_dt == 20:
                dt20_metrics_avg['aepe'].append(AEE.avg)
                dt20_metrics_avg['f1'].append(F1.avg)

        # Ouptut the results for a single scene
        out_str = 'Scene[{:02d}]: {:6s}'.format(ii_scene, cur_scene)
        if 10 in dt_list:
            out_str += '  dt10: (AEPE: {:.4f}, F1: {:.4f})  '.format(
                dt10_metrics_avg['aepe'][ii_scene], dt10_metrics_avg['f1'][ii_scene])
        if 20 in dt_list:
            out_str += '  dt20: (AEPE: {:.4f}, F1: {:.4f})  '.format(
                dt20_metrics_avg['aepe'][ii_scene], dt20_metrics_avg['f1'][ii_scene])

        out_str += 'Avg Time:  {:.4f}'.format(model_time.avg)

        log.info(out_str)
        
    out_str = 'Average of All the Scene  dt10: (AEPE: {:.4f}, F1: {:.4f})    dt20: (AEPE: {:.4f}, F1: {:.4f})'.format(
        sum(dt10_metrics_avg['aepe']) / len(dt10_metrics_avg['aepe']),
        sum(dt10_metrics_avg['f1'])   / len(dt10_metrics_avg['f1']),
        sum(dt20_metrics_avg['aepe']) / len(dt20_metrics_avg['aepe']),
        sum(dt20_metrics_avg['f1'])   / len(dt20_metrics_avg['f1']),
    )
    log.info(out_str)
    
    return



if __name__ == '__main__':
    ##########################################################################################################
    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if args.save_name == None:
        save_folder_name = 'b{:d}_{:s}'.format(args.batch_size, timestamp2)
    else:
        save_folder_name = 'b{:d}_{:s}_{:s}'.format(args.batch_size, timestamp2, args.save_name)

    save_root = osp.join(args.save_dir, timestamp1)
    save_path = osp.join(save_root, save_folder_name)
    make_dir(args.save_dir)
    make_dir(save_root)
    make_dir(save_path)
    make_dir(args.vis_path)
    make_dir(args.eval_vis)

    _log = init_logger(log_dir=save_path, filename=timestamp2+'.log')
    _log.info('=> will save everything to {:s}'.format(save_path))
    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations: \n' + cfg_str)

    train_writer = SummaryWriter(save_path)

    ##########################################################################################################
    ## Create model
    
    model = get_model(args)

    if args.pretrained:
        load_data = torch.load(args.pretrained)
        if 'optimizer' in load_data.keys():     # new save model
            network_data = load_data['model']
            optimizer_data = load_data['optimizer'] 
        else:                                   # old save model
            network_data = load_data

        _log.info('=> using pretrained flow model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        _log.info('=> train flow model from scratch')
        model.init_weights()
        _log.info('=> Flow model params: {:.6f}M'.format(model.num_parameters()/1e6))
        model = torch.nn.DataParallel(model).cuda()


    cudnn.benchmark = True


    
    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']
    assert(cfgopt['solver'] in ['Adam', 'SGD'])
    _log.info('=> settings {:s} solver'.format(cfgopt['solver']))
    
    param_groups = [{'params': model.module.parameters(), 'weight_decay': cfgmdl['flow_weight_decay']}]
    if cfgopt['solver'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfgopt['momentum'], cfgopt['beta']))
    elif cfgopt['solver'] == 'SGD':
        optimizer = torch.optim.SGD(param_groups, args.learning_rate, momentum=cfgopt['momentum'])

    if args.pretrained:
        if 'optimizer' in load_data.keys():
            optimizer.load_state_dict(optimizer_data)
            _log.info('=> using loaded optimizer')
            if args.start_epoch % 10 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * cfg['train']['decay_factor']

    scheduler = None
    
    ##########################################################################################################
    ## Dataset
    
    train_set = SpiftDataset(cfg=cfg)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last=True,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Define test loaders
    if args.eval:
        phm_scene_list = ['ball', 'cook', 'dice', 'doll', 'fan', 'hand', 'jump', 'poker', 'top']
        dt_list = [10, 20]
    else:
        phm_scene_list = ['ball', 'cook', 'dice', 'doll', 'fan', 'hand', 'jump', 'poker', 'top']
        dt_list = [10, 20]

    test_sets_dt10, test_sets_dt20 = get_phm_test_set(
        cfg=cfg, 
        scene_list=phm_scene_list, 
        dt_list=dt_list, 
        log=_log
    )

    test_loader_dt10 = [torch.utils.data.DataLoader(
        test_set, 
        drop_last=False,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    ) for test_set in test_sets_dt10]
    
    test_loader_dt20 = [torch.utils.data.DataLoader(
        test_set, 
        drop_last=False,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    ) for test_set in test_sets_dt20]

    test_loader_lists = [test_loader_dt10, test_loader_dt20]

    

    # Eval or Train
    if args.eval:
        with torch.no_grad():
            validate(cfg=cfg, 
                test_loader_lists=test_loader_lists, 
                scene_list=phm_scene_list, 
                dt_list=dt_list,
                model=model,
                log=_log
            )

    else:
        epoch = args.start_epoch
        while(True):
            train(cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scheduler=scheduler,
                log=_log,
                train_writer=train_writer,
            )
            epoch += 1

            # Save Model
            if epoch % 5 == 0:
                flow_model_save_name = '{:s}_epoch{:03d}.pth'.format(args.arch, epoch)

                save_file = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(save_file, osp.join(save_path, flow_model_save_name))

            # Validation
            # if (epoch % args.valid_freq == 0) or (epoch % cfg['loader']['n_epochs'] == 0):
            if epoch in [20, 40, 50]:
                with torch.no_grad():
                    validate(cfg=cfg,
                        test_loader_lists=test_loader_lists, 
                        scene_list=phm_scene_list, 
                        dt_list=dt_list,
                        model=model,
                        log=_log
                    )

            # Learning Rate Decay
            if epoch % cfg['train']['decay_freq'] == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * cfg['train']['decay_factor']

            if epoch >= cfg['loader']['n_epochs']:
                break