import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from torch.utils.data import SubsetRandomSampler
import copy
import time
import heapq
import glob

from model.loss import *
from model.utils import DataLoader
from model.Two_Path_Unet import SlowFastUnet
# from model.SlowFastUnet import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
from tqdm import tqdm

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="SelAD")
    parser.add_argument('--gpus', nargs='+',default="0", type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs for training')
    parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--t_length', type=int, default=9, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='shanghaitech', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='/datasets/', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    parser.add_argument('--divide', type = int, default = 50, help='divide training pool')
    parser.add_argument('--budget', type = int, default = 18, help = 'budget for active learning')
    parser.add_argument('--gd', type = int, default=1, help='parameter of gradient loss')
    parser.add_argument("--SelAD", default='True', action='store_false', help='operate selection or not')
    parser.add_argument('--gd_param',type = float, default='0.01', help='weight of gradient loss')
    parser.add_argument('--draw', type = bool, default=False, help = 'draw top-k')
    parser.add_argument('--topk', type = int, default= 1, help = "topk: 1, bottomk: 2, randomk: 3")

    return parser.parse_args()

def which_device(device):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if device is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(device)):
            gpus = gpus + device[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]
    device = torch.device('cuda')
    return device

def active_loader(init_dataset,dataset, label_indices, unlabel_indices,divide):
    """
    selection strategy
    random sample at first.
    """
    if init_dataset:
        # set indices for each pool
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        label_indices = random.sample(indices,int(len(dataset)/divide))
        unlabel_indices = list(set(indices) - set(label_indices))
        print("labeled and unlabeld length: {}, {}".format(len(label_indices), len(unlabel_indices)))

        label_sampler = SubsetRandomSampler(label_indices)
        unlabel_sampler = SubsetRandomSampler(unlabel_indices)

        # build pools
        train_batch = data.DataLoader(dataset, batch_size=args.batch_size, sampler=label_sampler,   
                                    shuffle=False, num_workers=4, drop_last=True)
        sample_batch = data.DataLoader(dataset, batch_size=args.test_batch_size, sampler=unlabel_sampler,   
                                    shuffle=False, num_workers=args.num_workers_test, drop_last=False)
        return label_indices, unlabel_indices, train_batch, sample_batch
    
    
    print(len(dataset))
    idx = random.sample(unlabel_indices, len(dataset)//(2*divide))
    # warning out of index
    for i in idx:
        unlabel_indices.remove(i)
    # add indices in labeled datapool
    for i in idx:
        label_indices.append(i)
    

    label_sampler = SubsetRandomSampler(label_indices)
    unlabel_sampler = SubsetRandomSampler(unlabel_indices)

    # update pools
    train_batch = data.DataLoader(dataset, batch_size=args.batch_size, sampler=label_sampler,   
                                    shuffle=False, num_workers=4, drop_last=True)
    sample_batch = data.DataLoader(dataset, batch_size=args.batch_size, sampler=unlabel_sampler,   
                                    shuffle=False, num_workers=args.num_workers_test, drop_last=False)
    print(len(label_indices), len(unlabel_indices))
    print(len(train_batch))
    return label_indices, unlabel_indices, train_batch, sample_batch

if __name__ == "__main__":
    args = get_args()
    print(args)

    cur_path=os.path.abspath('..')
    start_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    device = which_device(args.gpus)
    torch.manual_seed(2021)
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    
    train_folder = cur_path + args.dataset_path+args.dataset_type+"/training/frames"
    test_folder = cur_path + args.dataset_path+args.dataset_type+"/testing/frames"
    print(train_folder)
    # Loading dataset
    print("loading frames")
    # load total train set
    train_dataset = DataLoader(train_folder, transforms.Compose([
                transforms.ToTensor(),          
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
    if args.SelAD:
        # initial sampling
        dataset_size = len(train_dataset)
        # set indices for each pool
        indices = list(range(dataset_size))
        label_indices = random.sample(indices,dataset_size//args.divide)
        unlabel_indices = list(set(indices) - set(label_indices))

        # 这一步要把原来的标签保留下来，到时候才能在unlabeled下标里面对应
        selection_indices = random.sample(unlabel_indices, dataset_size//args.divide)

        print("labeled and unlabeld length: {}, {}".format(len(label_indices), len(unlabel_indices)))

        label_sampler = SubsetRandomSampler(label_indices)
        unlabel_sampler = SubsetRandomSampler(selection_indices)
        
        # build pools
        train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=label_sampler,shuffle=False, num_workers=4, drop_last=True)
        sample_batch = data.DataLoader(train_dataset, batch_size=args.test_batch_size, sampler=unlabel_sampler,shuffle=False, 
                                        num_workers=args.num_workers_test, drop_last=False)
        print("train size: {}".format(dataset_size))
    else:
        train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                                shuffle=False, num_workers=4, drop_last=True)

    # Model setting
    # model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
    model = SlowFastUnet()
    # params_decoder = list(model.decoder.parameters())
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr = args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
    # gradient_loss = Gradient_Loss(args.gd,args.c).cuda()
    model.cuda()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    loss_func_mse = nn.MSELoss(reduction='none')
    total_loss = []
    draw = False
    # Training
    m_pools = [F.normalize(torch.rand(( 10, 128), dtype=torch.float), dim=1).cuda(),
               F.normalize(torch.rand(( 10, 256), dtype=torch.float), dim=1).cuda(),
               F.normalize(torch.rand(( 10, 512), dtype=torch.float), dim=1).cuda(),
               F.normalize(torch.rand(( 10, 512), dtype=torch.float), dim=1).cuda()]
    for t in range(args.budget):
        draw = False
        add = 0
        if t == args.budget -1:
            add = args.epochs
        if t ==3:
            draw=True
        for epoch in range(args.epochs+add):
            labels_list = []
            model.train()
            now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
            print(now)
            for j,(imgs) in enumerate(tqdm(train_batch)):
                
                imgs = Variable(imgs).cuda()      # B, N*C, H, W
                outputs, fea, skip1, skip2, skip3, m_pools, loss_mem = model.forward(imgs[:,0:24,],m_pools,True)
                # print(fea.size())            
               
                optimizer.zero_grad()
                # gd_loss=gradient_loss(outputs,imgs[:,24:])
                mse_loss = torch.mean(loss_func_mse(outputs, imgs[:,24:]))
                loss = mse_loss+loss_mem
                # loss = args.gd_param*gd_loss+mse_loss
                loss.backward(retain_graph=True)
                optimizer.step()   
            
            scheduler.step()
            total_loss.append([loss.item(),loss_mem.item()])
            print('----------------------------------------')
            print('Epoch:', epoch+1)
            print('Loss: Prediction {:.6f}, Memory {:.6f}'.format(loss.item(),loss_mem.item()))
            print('----------------------------------------')
        
        # inference procedure for sample selection
        if args.SelAD:
            print("Start inference")
            model.eval()
            score = []
            for _,(imgs) in enumerate(tqdm(sample_batch)):
                imgs = Variable(imgs).cuda()
                
                outputs, fea, skip1, skip2, skip3, m_pools, loss_mem = model.forward(imgs[:,0:24,],m_pools,True)  # for reconstruction
                
                # gd_loss = gradient_loss(outputs, imgs[:,24:]).item()
                mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*8:]+1)/2)).item()
                diff = mse_imgs
                # diff = mse_imgs*gd_loss
                score.append(psnr(diff))
            score = anomaly_score_list(score)
            # print(np.max(score))
            #topk = heapq.nsmallest(dataset_size//(2*args.divide), range(len(score)), score.__getitem__)
            if args.topk == 1:
                topk = heapq.nsmallest(dataset_size//(2*args.divide), range(len(score)), score.__getitem__)
                idx = [selection_indices[i] for i in topk]
            elif args.topk == 2:
                topk = heapq.nsmallest(dataset_size//(2*args.divide), range(len(score)), score.__getitem__)
                idx = [selection_indices[i] for i in topk]
            elif args.topk == 3:
                idx = [selection_indices[i] for i in topk] 
            else:
                print("Invalid selection strategy, reset to top-k")
                topk = heapq.nsmallest(dataset_size//(2*args.divide), range(len(score)), score.__getitem__)
                idx = [selection_indices[i] for i in topk]
          
            # print(idx)
            # print(heapq.nlargest(3,score))
            for i in idx:
                unlabel_indices.remove(i)
            # add indices in labeled datapool
            for i in idx:
                label_indices.append(i)  

            # resample
            selection_indices = random.sample(unlabel_indices, dataset_size//args.divide)
            label_sampler = SubsetRandomSampler(label_indices)
            unlabel_sampler = SubsetRandomSampler(selection_indices)

            # update pools
            train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=label_sampler,   
                                            shuffle=False, num_workers=4, drop_last=True)
            sample_batch = data.DataLoader(train_dataset, batch_size=args.test_batch_size, sampler=unlabel_sampler,   
                                            shuffle=False, num_workers=args.num_workers_test, drop_last=False)
            print('*'*40)
            print('selection '+str(t)+' finished')
            print('label / unlabel: {} / {}'.format(len(label_indices), len(unlabel_indices)))
            print('*'*40)
    end_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    print('Training is finished')
    print('Stat: {}. End time: {}.'.format(start_time, end_time))
    np.save("total_loss"+args.dataset_type+".npy",total_loss)
    
    # Loading test dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                transforms.ToTensor(),            
                ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                                shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    # label processing
    labels = np.load('../data/frame_labels_'+args.dataset_type+'.npy')
    if args.dataset_type == 'shanghaitech':
        labels = np.expand_dims(labels, 0)
        print(args.dataset_type)
    print("len labels: {}.".format(len(labels)))

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    print('len video: {}'.format(len(videos_list)))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort(key = lambda x: int(x.split('/')[-1][:-4]))
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        labels_list = np.append(labels_list, labels[0][8+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    fea_list = []
    model.eval()
    loss_func_mse = nn.MSELoss(reduction='none')
    gradient_loss=Gradient_Loss(args.gd,args.c).cuda()

    for k,(imgs) in enumerate(tqdm(test_batch)):
        draw = False
        if k == label_length-8*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            # print(videos[videos_list[video_num].split('/')[-1]])
        
        imgs = Variable(imgs).cuda()

        # outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:,0:3*4], m_items_test, False)
        outputs, fea, skip1, skip2, skip3, m_pools, loss_mem = model.forward(imgs[:,0:24,],m_pools,True)
        # if k == 150:
        #     np.save("fea150.npy", fea.cpu().detach().numpy())
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*8:]+1)/2)).item()
        # gd_loss = gradient_loss(outputs, imgs[:,24:]).item()
        # print(gd_loss)
        # print((outputs[0]-imgs[0,3*8:]).shape)
        diff = mse_imgs

        psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(diff))
    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),anomaly_score_list(psnr_list[video_name]),1)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)
    # np.save("train2eval.npy", anomaly_score_total_list)
    accuracy = AUC(anomaly_score_total_list, np.expand_dims(labels_list, 0))
    print('Training time:')
    print('Stat: {}. End time: {}.'.format(start_time, end_time))
    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%')
