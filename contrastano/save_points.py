import torch
import torch.nn.functional as F
import utils
import numpy as np
from torch.autograd import Variable
import scipy.io as sio

def save_points(test_loader, model, device, args):
    result = {}
    for i, data in enumerate(test_loader):
        feature, data_video_name = data
        # print("feashape:{}, video name:{}".format(feature.shape,data_video_name))
        
        feature = feature.to(device)
        with torch.no_grad():
            # if args.model_name == 'model_lstm':
            #     _, element_logits = model(feature, seq_len=None, is_training=False)
            # else:
            feas, _ = model(feature,is_training=False)
            
        feas = feas.cpu().data.numpy()
        # print(element_logits)
        # element_logits = F.softmax(element_logits, dim=2)[:, :, 1].cpu().data.numpy()
        # element_logits = element_logits.cpu().data.numpy()
        result[data_video_name[0]] = feas
    np.save("feas.npy",result)




