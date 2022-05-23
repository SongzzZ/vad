import torch
import torch.nn as nn

class Gradient_Loss(nn.Module):
    def __init__(self,alpha,channels):
        super(Gradient_Loss,self).__init__()
        self.alpha=alpha
        filter=torch.FloatTensor([[-1.,1.]]).cuda()

        self.filter_x = filter.view(1,1,1,2).repeat(1,channels,1,1)
        self.filter_y = filter.view(1,1,2,1).repeat(1,channels,1,1)


    def forward(self, gen_frames,gt_frames):


        # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
        # neg=-1*pos
        # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
        # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
        gen_frames_x=nn.functional.pad(gen_frames,(1,0,0,0))
        gen_frames_y=nn.functional.pad(gen_frames,(0,0,1,0))
        gt_frames_x=nn.functional.pad(gt_frames,(1,0,0,0))
        gt_frames_y=nn.functional.pad(gt_frames,(0,0,1,0))

        gen_dx=nn.functional.conv2d(gen_frames_x,self.filter_x)
        gen_dy=nn.functional.conv2d(gen_frames_y,self.filter_y)
        gt_dx=nn.functional.conv2d(gt_frames_x,self.filter_x)
        gt_dy=nn.functional.conv2d(gt_frames_y,self.filter_y)

        grad_diff_x=torch.abs(gt_dx-gen_dx)
        grad_diff_y=torch.abs(gt_dy-gen_dy)

        return torch.mean(grad_diff_x**self.alpha+grad_diff_y**self.alpha)