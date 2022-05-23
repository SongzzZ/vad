import numpy as np
import os
import sys
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from .memory import *

sys.path.append('../')
from utils import draw_features

class Decoder(torch.nn.Module):
    def __init__(self, t_length = 8, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128,n_channel,64)
        
        
        
    def forward(self, x, skip1, skip2, skip3):
        
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim = 1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim = 1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim = 1)
        
        output = self.moduleDeconv1(cat2)

        # if draw:
        #     draw_features(3,1,output.cpu().detach().numpy(),"/home/zhaosong/MNAD1/figures/output_total.png")
        #     # print(output.shape)
        #     # print(tensorUpsample2.shape)
        #     # print(tensorUpsample3.shape)
        #     print("output: {}".format(output.shape))
 
        return output

class SlowFastUnet(torch.nn.Module):
    def __init__(self, t_length=8, n_channel=3):
        super(SlowFastUnet, self).__init__()

        def SBasic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def SBasic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        def FBasic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),)
        
        def FBasic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),)
        
        
        def lateral(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False), )
        
        def extraction(intInput, intOutput):
            return torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.SlowConv1 = SBasic(n_channel*t_length//2, 64)
        self.SlowPool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.SlowConv2 = SBasic(128, 128)
        self.SlowPool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.SlowConv3 = SBasic(256, 256)
        self.SlowPool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.SlowConv4 = SBasic_(512, 512)

        self.lateral1 = lateral(64,64)
        self.lateral2 = lateral(128,128)
        self.lateral3 = lateral(256,256)
        self.FastConv1 = FBasic(n_channel*t_length, 64)
        self.FastPool1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.FastConv2 = FBasic(64, 128)
        self.FastPool2 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.FastConv3 = FBasic(128, 256)
        self.FastPool3 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.FastConv4 = FBasic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        self.mp1 = Memory(10, 64,  64, 0.1, 0.1)
        self.mp2 = Memory(10,128, 128, 0.1, 0.1)
        self.mp3 = Memory(10,256, 256, 0.1, 0.1)
        self.mp4 = Memory(10,512, 512, 0.1, 0.1)
        self.decoder = Decoder(t_length, n_channel)
        

        
    def SPath(self, x, lateral, keys, train):
        tensorConv1 = self.SlowConv1(x)
        tensorPool1 = self.SlowPool1(tensorConv1)                      # [8, 64, 128, 128]
        # mempool = self.mempool1(tensorPool1)                         # [8, 64,  16,  16]
        tensorcat1 = torch.cat([tensorPool1, lateral[0]], dim=1)       # [8, 128, 128, 128]
        updated_cat1, keys1, softmax_score_query1, softmax_score_memory1, loss_mem1 = self.mp1(tensorcat1, keys[0], train)
        # print('cat1 size: {}'.format(tensorcat1.size()))

        tensorConv2 = self.SlowConv2(tensorcat1)                       
        tensorPool2 = self.SlowPool2(tensorConv2)                      # [8, 128, 64, 64]
        tensorcat2 = torch.cat([tensorPool2, lateral[1]], dim=1)
        updated_cat2, keys2, softmax_score_query2, softmax_score_memory2, loss_mem2 = self.mp2(tensorcat2, keys[1], train)

        tensorConv3 = self.SlowConv3(tensorcat2)
        tensorPool3 = self.SlowPool3(tensorConv3)
                               # [8, 256, 32, 32]
        # mempool3 = self.mempool3(tensorPool3)                           # [8, 256, 16, 16]
        tensorcat3 = torch.cat([tensorPool3, lateral[2]], dim=1)
        updated_cat3, keys3, softmax_score_query3, softmax_score_memory3, loss_mem3 = self.mp3(tensorcat3, keys[2], train)
        # tensormb3 = torch.einsum('bmhw, mf',tensorcat3,self.mb3)

        tensorConv4 = self.SlowConv4(tensorcat3)
        updated_fea, keys4, softmax_score_query4, softmax_score_memory4, loss_mem4= self.mp4(tensorConv4, keys[3], train)
        fea = torch.cat([tensorConv4,updated_fea], dim =1)
        updated_keys=[keys1, keys2, keys3, keys4]
        losses =  loss_mem4# loss_mem1+loss_mem2+loss_mem3+loss_mem4

        # mempool4 = self.mempool4(tensorConv4)                            # [8, 512, 16,16]
        # print('mempool4 size: {}'.format(mempool4.size())) 

        return fea, tensorConv1, tensorConv2, tensorConv3, updated_keys, losses

    def FPath(self, x):
        lateral = []
        tensorConv1 = self.FastConv1(x)
        # print('conv1: {}'.format(tensorConv1.shape))
        tensorPool1 = self.FastPool1(tensorConv1)
        lateral1 = self.lateral1(tensorPool1)
        lateral.append(lateral1)
        # print('pool1: {}'.format(tensorPool1.shape))

        tensorConv2 = self.FastConv2(tensorPool1)
        # print('conv2: {}'.format(tensorConv2.shape))
        tensorPool2 = self.FastPool2(tensorConv2)
        lateral2 = self.lateral2(tensorPool2)
        lateral.append(lateral2)
        # print('pool2: {}'.format(tensorPool2.shape))

        tensorConv3 = self.FastConv3(tensorPool2)
        # print('conv3: {}'.format(tensorConv3.shape))
        tensorPool3 = self.FastPool3(tensorConv3)
        #print('pool3: {}'.format(tensorPool3.shape))
        lateral3 = self.lateral3(tensorPool3)
        lateral.append(lateral3)
        

        tensorConv4 = self.FastConv4(tensorPool3)
        # print('conv4: {}'.format(tensorConv4.shape))
        
        return tensorConv4, lateral
        
    def forward(self, x, keys, train=True):
        # xs = torch.cat([x[:,:3], x[:,6:9,], x[:,12:15], x[:,18:21]], dim=1)
        # xs = torch.cat([x[:,3:6], x[:,9:12], x[:,15:18], x[:,21:24]], dim=1)
        # print(xs.size())
        _, lateral = self.FPath(x)
        
        updated_fea, skip1, skip2, skip3, updated_keys, losses= self.SPath(x[:,::2,:,:], lateral, keys, train)
        # if draw:
        #     np.save('f1.npy',f1.cpu().detach().numpy())
        #     np.save('lateral2.npy',lateral[2].cpu().detach().numpy())
        #     np.save('tp1.npy',tp1.cpu().detach().numpy())
        #     np.save('tp3.npy',tp3.cpu().detach().numpy())
        # fea_new = torch.einsum('bmhw,mf->bfhw',f1,self.mb4)
        # print(fea_new.shape)
        output = self.decoder(updated_fea, skip1, skip2, skip3)

        return output, updated_fea, skip1, skip2, skip3, updated_keys, losses

# def multiply(x): #to flatten matrix into a vector 
#     return functools.reduce(lambda x,y: x*y, x, 1)

# if __name__ =="__main__":
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"]= 'cuda:0'
#     device = torch.device('cuda')
#     input = torch.autograd.Variable(torch.rand(8, 27, 256, 256)).cuda()
#     model = SlowFastUnet().cuda()
#     output, f1 = model.forward(input[:,:24,:])
#     batch_size, dims,h,w = f1.size() # b X d X h X w
#     query = F.normalize(f1, dim=1)
#     print(query.size())
#     query = query.permute(0,2,3,1) # b X h X w X d
#     print(query.size())
