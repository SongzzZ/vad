import xxlimited
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import numpy as np
import random
from scipy import ndimage
from utils import fill_context_mask
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from selfattention import ScaledDotProductAttention

random.seed(2022)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)  # input n_feature, output n_feature
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.sa = ScaledDotProductAttention(d_model=2048, d_k=512, d_v=512, h=4)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))

        # x = F.relu(self.sa(inputs,inputs,inputs))   # X_i with self attention
        
        if is_training:
            x = self.dropout(x)
        # print(self.sigmoid(self.classifier(x)).shape) #X_i^FC
        return x, self.sigmoid(self.classifier(x))  #1/(1+exp(FC_AR))

class Model_SAD(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_SAD, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)  # input n_feature, output n_feature
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        # self.sa = ScaledDotProductAttention(d_model=2048, d_k=512, d_v=512, h=4)
        self.apply(weights_init)
        self.ce_loss = nn.CrossEntropyLoss()
    
    # do not work, due to the problem of zero, thus remove the ce_loss
    # def NCE(self, q, center, neg, T=0.07):
    #     # input query  [2048]
    #     q = nn.functional.normalize(q, dim=0)
    #     # positive sample [2048]
    #     center = nn.functional.normalize(center, dim=0)
    #     neg = nn.functional.normalize(neg, dim=0)
    #     # negative sample
    #     l_pos = torch.dot(q,center)
    #     print(l_pos)
    #     l_neg = torch.sum(torch.matmul(neg,center),dim=0)
    #     print(l_neg)

    #     l_pos_t = torch.exp(l_pos / T)   
    #     l_neg_t = torch.exp(l_neg / T)     
    
    #     loss_partial = -torch.log(l_pos_t / (l_neg_t+l_pos_t))        # 2*bs
    #     return loss_partial
    def NCE(self, q, k, neg, T=0.07):
        # print(neg.shape)
        q = nn.functional.normalize(q, dim=1)
        # print(q.shape)
        k = nn.functional.normalize(k, dim=1)
        # print(k.shape)
        neg = nn.functional.normalize(neg, dim=1).unsqueeze(-1)
        # print(neg.shape)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # print(l_pos.shape)
        l_neg = torch.einsum('nc,kcn->nk', [q, neg])
        # print(l_neg.shape)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        # print(logits.shape)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # print(labels.shape)
        loss = self.ce_loss(logits, labels)
        return loss

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))  # B * T * C
        # contrastive = False
        # eroding = True
        # top_anomaly=False
        # x = F.relu(self.sa(inputs,inputs,inputs))   # X_i with self attention
        if is_training:
            x = self.dropout(x)
            # output=self.sigmoid(self.classifier(x))
            
            # normal_num = 0
            # center_loss = torch.zeros(0).to(device)
            # sep_loss = torch.zeros(0).to(device)
            # real_size = int(x.shape[0])
            # center=torch.zeros(x.shape[2]).to(device)
            # normal_type = torch.zeros(1,x.shape[2]).to(device)
            # ano_type=x[random.randint(0,real_size-1)][random.randint(0,int(x.shape[1])-1)].unsqueeze(0)
     
            # for i in range(real_size):
            #     if labels[i] == 0:
            #         center = torch.mean(x[i],dim=0) 
            #         idx,_=torch.topk(output[i].squeeze(), 3, dim=0,largest=False)
            #         for j in idx.data.cpu().detach().numpy():
            #             normal_type=torch.cat((normal_type,x[i][np.int(j)].unsqueeze(0)))
                    
            #         normal_num+=1
            #     # find top-k in abnormal clips, considering other ways to find the clip
            #     # eroding to find contrastive part
            #     elif labels[i] == 1:
            #         if eroding:
            #             score_np = output[i].squeeze().cpu().detach().numpy()
            #             score_median = np.median(score_np, 0, keepdims=True)
            #             score_bin = np.where(score_np > score_median, 1, 0)
            #             #print("TOBE dilated and eroded:{}".format(score_bin))

            #             erosion_m1 = ndimage.binary_erosion(score_bin, structure=np.ones(2)).astype(score_bin.dtype)
            #             # print("erosion_m1:{}".format(erosion_m1))
            #             dilation_m = ndimage.binary_dilation(erosion_m1, structure=np.ones(4)).astype(score_bin.dtype)
            #             #print("dilation_m:{}".format(dilation_m))

            #             erosion_m = ndimage.binary_erosion(dilation_m, structure=np.ones(6)).astype(score_bin.dtype)
            #             #print("erosion_m:{}".format(erosion_m))
            #             for j in range(len(erosion_m)):
            #                 if erosion_m[j]==1:
            #                     ano_type=torch.cat((ano_type,x[i][np.int(j)].unsqueeze(0)))

            #         elif top_anomaly:
            #             idx,_=torch.topk(output[i].squeeze(), 3, dim=0)
            #             for j in idx.data.cpu().detach().numpy():
            #                 ano_type=torch.cat((ano_type,x[i][np.int(j)].unsqueeze(0)))
                        
            # center = center/normal_num

            # for i in range(real_size):
            #     if labels[i]==0:
            #         # print(torch.dot(x[i][0],center))
            #         center_loss = torch.mean(torch.sum((x[i] - center) ** 2, dim=1))  
            #         # for j in x[i]:
            #         #     center_loss+=self.NCE(j.unsqueeze(0),center.unsqueeze(0),ano_type)          
            
            # # abnormal clips guided loss
            # # sep_loss = torch.mean(torch.sum((ano_type - center+0.01) ** (-2), dim=1))
                
            # sep_loss = self.NCE(torch.mean(ano_type,dim=0).unsqueeze(0),torch.mean(ano_type,dim=0).unsqueeze(0),normal_type)
            # if contrastive: 
            #     loss_contra = self.NCE(ano_type[0],center,normal_type)
            #     print("contrastive loss:{}".format(loss_contra))


            # center_loss=center_loss
            # sep_loss=sep_loss*5e-8/(1+real_size)
            #sep_loss=sep_loss
            
            # print(self.sigmoid(self.classifier(x)).shape) #X_i^FC
            return x, self.sigmoid(self.classifier(x))#1/(1+exp(FC_AR))
        return x, self.sigmoid(self.classifier(x))

# introduce scores as supervision
class Model_SAD_Bad(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_SAD_Bad, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)  # input n_feature, output n_feature
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        # self.sa = ScaledDotProductAttention(d_model=2048, d_k=512, d_v=512, h=4)
        self.apply(weights_init)

    def forward(self, inputs, device,labels,  is_training=True):
        x = F.relu(self.fc(inputs))  # B * T * C
        # x = F.relu(self.sa(inputs,inputs,inputs))   # X_i with self attention
        if is_training:
            x = self.dropout(x)
            normal_num = 0
            real_size = int(x.shape[0])
            center=torch.zeros(x.shape[2]).to(device)
            
            for i in range(real_size):
                if labels[i] == 0:
                    center += torch.mean(x[i],dim=0)
                    normal_num+=1
            # print('************************')
            # print(center)
            center = center/normal_num
            # print(center)

            for i in range(real_size):
                if labels[i]==0:
                    center_loss = torch.mean(torch.sum((x[i] - center) ** 2, dim=1))
                elif labels[i]==1:
                    sep_loss = torch.mean(torch.sum((x[i] - center+0.01) ** (-2), dim=1))
            center_loss=center_loss*0.5/(1+real_size)
            sep_loss=sep_loss*1e-7/(1+real_size)
            # print(center_loss.item(),sep_loss.item())
        
            # print(self.sigmoid(self.classifier(x)).shape) #X_i^FC
            return x, self.sigmoid(self.classifier(x)), center_loss+sep_loss  #1/(1+exp(FC_AR))
        return x, self.sigmoid(self.classifier(x))

class Filter_Module(nn.Module):
    def __init__(self, len_feature):
        super(Filter_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                      stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1,
                      stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)
        return out


class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.drop_out(out)
        out = self.conv_3(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out


class BaS_Net(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments):
        super(BaS_Net, self).__init__()
        self.filter_module = Filter_Module(len_feature)
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.num_segments = num_segments
        self.k = num_segments // 8

    def forward(self, x):
        fore_weights = self.filter_module(x)
        x_supp = fore_weights * x

        cas_base = self.cas_module(x)
        cas_supp = self.cas_module(x_supp)

        score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=1)
        score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=1)

        score_base = self.softmax(score_base)
        score_supp = self.softmax(score_supp)

        return score_base, cas_base, score_supp, cas_supp, fore_weights


#
# class Model_single(torch.nn.Module):
#     def __init__(self, n_feature):
#         super(Model_single, self).__init__()
#         self.fc = nn.Linear(n_feature, n_feature)
#         self.classifier = nn.Linear(n_feature, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.7)
#         self.apply(weights_init)
#
#     def forward(self, inputs, is_training=True):
#         x = F.relu(self.fc(inputs))
#         if is_training:
#             x = self.dropout(x)
#         return x, self.classifier(x), self.sigmoid(self.classifier(x))


class Model_mean(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_mean, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)
        self.mean_pooling = nn.AvgPool2d((3, 1))
        # self.weight_conv1 = nn.Conv2d(n_channels, out_channels, kernel_size, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs)).permute(0, 2, 1).unsqueeze(2)
        x_2 = F.relu(self.conv2(inputs)).permute(0, 2, 1).unsqueeze(2)
        x_3 = F.relu(self.conv3(inputs)).permute(0, 2, 1).unsqueeze(2)
        x = torch.cat((x_1, x_2, x_3), dim=2)
        x = self.mean_pooling(x)
        # x = x_3 + x_2
        # x = F.relu(self.conv_b2(x))
        # x = x_1 + x
        # x = F.relu(self.conv_b1(x))
        x = x.squeeze(2)
        if is_training:
            x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))

class Model_sequence(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_sequence, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = x_3 + x_2
        x = F.relu(self.conv_b2(x))
        x = x_1 + x
        x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x, self.sigmoid(self.classifier(x))

class Model_concatcate(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_concatcate, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        # self.conv_b1 = nn.Conv1d(in_channels=n_feature * 3, out_channels=n_feature, kernel_size=1, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')
        # self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
        #          padding=0, dilation=1, groups=1,
        #          bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc(x))

        # x = x_3 + x_2
        # x = F.relu(self.conv_b2(x))
        # x = x_1 + x
        # x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)

        return x, self.sigmoid(self.classifier(x))

# class Model_attention(nn.Module):
#     def __init__(self, args):
#         super(Model_attention, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.fill_context_mask = fill_context_mask
#         self.args = args
#
#     def forward(self, final_features, element_logits, seq_len, labels):
#         seq_len_list = seq_len.tolist()
#         for i in range(len(element_logits)):
#             if labels[i] == 0:
#                 element_logits[i] = 1 - element_logits[i]
#         element_logits = torch.transpose(element_logits, 2, 1)
#         mask = self.fill_context_mask(mask=element_logits.clone(), sizes=seq_len_list, v_mask=float('-inf'), v_unmask=0)
#         attention_logits = element_logits + mask
#         if self.args.attention_type == 'softmax':
#             attention_logits = F.softmax(attention_logits, dim=2)
#         elif self.args.attention_type == 'sigmoid':
#             attention_logits = self.sigmoid(attention_logits)
#         else:
#             raise ('attention_type is out of option')
#         M = torch.bmm(attention_logits, final_features).squeeze(1)
#
#         return M

class model_lstm(torch.nn.Module):
    def __init__(self, n_feature):
        super(model_lstm, self).__init__()
        self.bidirectlstm = nn.LSTM(
            input_size=n_feature,
            hidden_size=n_feature,
            num_layers=1,
            batch_first=True)
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.7)

    def forward(self, inputs, seq_len, is_training=True):
        if is_training:
            seq_len_list = seq_len.tolist()
            x = pack_padded_sequence(input=inputs, lengths=seq_len_list, batch_first=True, enforce_sorted=False)
            x, _ = self.bidirectlstm(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
            # x = self.dropout(x)
        else:
            x, _ = self.bidirectlstm(inputs)
        return x, self.sigmoid(self.classifier(x))

def model_generater(model_name, feature_size):
    if model_name == 'model_single':
        model = Model_single(feature_size)  # for anomaly detection, only one class, anomaly, is needed.
    elif model_name == 'model_sad':
        model = Model_SAD(feature_size)
    elif model_name == 'model_mean':
        model = Model_mean(feature_size)
    elif model_name == 'model_sequence':
        model = Model_sequence(feature_size)
    elif model_name == 'model_concatcate':
        model = Model_concatcate(feature_size)
    elif model_name == 'model_lstm':
        model = model_lstm(feature_size)
    elif model_name == 'model_bas':
        model = BaS_Net(feature_size)
    else:
        raise ('model_name is out of option')
    return model