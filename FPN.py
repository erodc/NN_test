import torch
import torch.nn.functional as F
import torch.nn as nn


class FPN(nn.Module):

    def __init__(self,in_channel_list,out_channel):
        super(FPN, self).__init__()
        self.inner_layer=[]
        self.out_layer=[]
        for in_channel in in_channel_list:
            self.inner_layer.append(nn.Conv2d(in_channel,out_channel,1))
            self.out_layer.append(nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1))
        # self.upsample=nn.Upsample(size=, mode='nearest')

    def forward(self,x):
        head_output=[]
        corent_inner=self.inner_layer[-1](x[-1])
        head_output.append(self.out_layer[-1](corent_inner))
        for i in range(len(x)-2,-1,-1):
            pre_inner=corent_inner
            corent_inner=self.inner_layer[i](x[i])
            size=corent_inner.shape[2:]
            pre_top_down=F.interpolate(pre_inner,size=size)
            add_pre2corent=pre_top_down+corent_inner
            head_output.append(self.out_layer[i](add_pre2corent))
        return list(reversed(head_output))


if __name__ == '__main__':
    fpn = FPN([10,20,30],5)
    x = []
    x.append(torch.rand(1, 10, 64, 64))
    x.append(torch.rand(1, 20, 16, 16))
    x.append(torch.rand(1, 30, 8, 8))
    c = fpn(x)
    print(c)
