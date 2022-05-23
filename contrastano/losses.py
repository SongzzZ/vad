from telnetlib import DEBUGLEVEL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import random
import heapq
random.seed(2022)

# import options
# from video_dataset_anomaly_balance_sample import dataset # For anomaly
# from torch.utils.data import DataLoader
# import math
# from utils import fill_context_mask, median

mseloss = torch.nn.MSELoss(reduction='mean')
mseloss_vector = torch.nn.MSELoss(reduction='none')
ce_loss = nn.CrossEntropyLoss()
binary_CE_loss = torch.nn.BCELoss(reduction='mean')
binary_CE_loss_vector = torch.nn.BCELoss(reduction='none')

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

def hinger_loss(anomaly_score, normal_score):
    return F.relu((1 - anomaly_score + normal_score))

def init_center_c(train_loader, net, dev, eps=0.1,):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=dev)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _, _, _ = data
            inputs = inputs.to(dev)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def NCE(q, k, neg, T=0.07):
    neg = neg.unsqueeze(-1)
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,kcn->nk', [q, neg])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= T
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss = ce_loss(logits, labels)
    return loss

def sad_logit(element_logits, labels, device): 
    normal_num = 0
    center_loss = torch.zeros(0).to(device)
    sep_loss = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0]) # incase of uncomplete batch
    normal_type = torch.zeros(1,element_logits.shape[2]).to(device)
    ano_val, ano_idx = torch.max(element_logits,dim=0) 
    ano_type=ano_val[torch.min(ano_idx)].unsqueeze(0)
    bool_print = True
     
    for i in range(real_size):
        if labels[i] == 0:
            center = torch.mean(element_logits[i],dim=0) 
            idx,_=torch.topk(element_logits[i].squeeze(), 3, dim=0,largest=False)
            for j in idx.data.cpu().detach().numpy():
                normal_type=torch.cat((normal_type,element_logits[i][np.int(j)].unsqueeze(0)))
                    
            normal_num+=1
        # find top-k in abnormal clips, considering other ways to find the clip
        # eroding to find contrastive part
        elif labels[i] == 1:
            score_np = element_logits[i].squeeze().cpu().detach().numpy()
            # # median
            # score_median = np.median(score_np, 0, keepdims=True)
            # score_bin = np.where(score_np > score_topk, 1, 0)
            # top-k
            score_topk=heapq.nlargest(len(score_np)//5,score_np)[-1]
            score_bin = np.where(score_np > score_topk, 1, 0)
        
#            erosion_m1 = ndimage.binary_erosion(score_bin, structure=np.ones(2)).astype(score_bin.dtype)
#            dilation_m = ndimage.binary_dilation(erosion_m1, structure=np.ones(4)).astype(score_bin.dtype)
#            erosion_m = ndimage.binary_erosion(dilation_m, structure=np.ones(6)).astype(score_bin.dtype)
#
#            if bool_print:
#                print(len(score_np))
#                print("score np:")
#                print(score_np)
#                print("score bin:")
#                print(score_bin)
#                print("score erosion:")
#                print(erosion_m1)
#                print("score dilation:")
#                print(erosion_m)
#                bool_print=False

# use score bin
    for j in range(len(score_bin)):
        if score_bin[j]==1:
            ano_type=torch.cat((ano_type,element_logits[i][np.int(j)].unsqueeze(0)))                    
    center = center/normal_num
    # for i in range(real_size):
    #     if labels[i]==0:
    #         # print(torch.dot(x[i][0],center))
    #         center_loss = torch.mean(torch.sum((element_logits[i] - center) ** 2, dim=1))  
            # for j in x[i]:
            #     center_loss+=self.NCE(j.unsqueeze(0),center.unsqueeze(0),ano_type)          
            
    # abnormal clips guided loss
    # sep_loss = torch.mean(torch.sum((ano_type - center+0.01) ** (-2), dim=1))
    center_loss = NCE(torch.mean(normal_type,dim=0).unsqueeze(0),torch.mean(normal_type,dim=0).unsqueeze(0),ano_type)           
    sep_loss = NCE(torch.mean(ano_type,dim=0).unsqueeze(0),torch.mean(ano_type,dim=0).unsqueeze(0),normal_type)

    return (center_loss+sep_loss)/(real_size+1)  #1/(1+exp(FC_AR))

def sad_loss(element_logits, labels, device):
    print('element_logits shape:{}'.format(element_logits))
    eps = 1e-6
    normal_num = 0
    norm_loss = torch.zeros(0).to(device)

    center_sum = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    for i in range(real_size):
        if labels[i] == 0:
            center_sum = torch.sum(element_logits[i],dim=0)
            normal_num+=1
    center = center_sum/normal_num

    # # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    # center[(abs(center) < 1e-8) & (center < 0)] = -eps
    # center[(abs(center) < 1e-8) & (center > 0)] = eps
    # print(center)

    for i in range(real_size):
        norm_loss = torch.cat((norm_loss, torch.var(element_logits[i]).unsqueeze(0)))
        if labels[i] == 0:
            center_loss = torch.mean(torch.sum((element_logits[i] - center) ** 2, dim=1))
        elif labels[i] == 1:
            sep_loss = torch.mean(torch.sum((element_logits[i] - center+eps) ** (-2), dim=1))
    # print(center_loss.item(),sep_loss.item())
    center_loss = normal_num*center_loss/real_size
    sep_loss = (real_size+1-normal_num)*sep_loss/real_size
    norm_loss = torch.mean(norm_loss, dim=0)
    # print(center_loss.item(),sep_loss.item(),norm_loss.item())

    return center_loss+sep_loss+norm_loss
   
def center_loss(element_logits, labels, device):
    
    normal_num = 0
    center_sum = torch.zeros(0).to(device)

    real_size = int(element_logits.shape[0])
    for i in range(real_size):
        if labels[i] == 0:
            center_sum = torch.sum(element_logits[i],dim=0)
            normal_num+=1
    center = center_sum/normal_num
    
    for i in range(real_size):
        if labels[i] == 0:
            center_loss = torch.mean(torch.sum((element_logits[i] - center) ** 2, dim=1))
    center_loss=center_loss/normal_num

    return center_loss

def normal_smooth(element_logits, labels, device):
    """
    :param element_logits:
    :param seq_len:
    :param batch_size:
    :param labels:
    :param device:
    :param loss:
    :return:
    """

    normal_smooth_loss = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])

    for i in range(real_size):
        if labels[i] == 0:
            normal_smooth_loss = torch.cat((normal_smooth_loss, torch.var(element_logits[i]).unsqueeze(0)))
    normal_smooth_loss = torch.mean(normal_smooth_loss, dim=0)
    
    return normal_smooth_loss

def KMXMILL_individual(element_logits,
                       seq_len,
                       labels,
                       device,
                       loss_type='CE',
                       args=None):

    """
    :param element_logits:
    :param seq_len:
    :param batch_size:
    :param labels:
    :param device:
    :param loss:
    :return:
    """
    # [train_video_name, start_index, len_index] = stastics_data
    k = np.ceil(seq_len/(args.k-1)).astype('int32')
    instance_logits = torch.zeros(0).to(device)
    real_label = torch.zeros(0).to(device)
    real_size = int(element_logits.shape[0])
    # because the real size of a batch may not equal batch_size for last batch in a epoch
    for i in range(real_size):
        tmp, tmp_index = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        # top_index = np.zeros(len_index[i].numpy())
        # top_predicts = np.zeros(len_index[i].numpy())
        # top_index[tmp_index.cpu().numpy() + start_index[i].numpy()] = 1
        # if train_video_name[i][0] in log_statics:
        #     log_statics[train_video_name[i][0]] = np.concatenate((log_statics[train_video_name[i][0]], np.expand_dims(top_index, axis=0)),axis=0)
        # else:
        #     log_statics[train_video_name[i][0]] = np.expand_dims(top_index, axis=0)
        instance_logits = torch.cat((instance_logits, tmp), dim=0)
        if labels[i] == 1:
            real_label = torch.cat((real_label, torch.ones((int(k[i]), 1)).to(device)), dim=0)
        else:
            real_label = torch.cat((real_label, torch.zeros((int(k[i]), 1)).to(device)), dim=0)
    if loss_type == 'CE':
        milloss = binary_CE_loss(input=instance_logits, target=real_label)
        return milloss
    elif loss_type == 'MSE':
        milloss = mseloss(input=instance_logits, target=real_label)
        return milloss