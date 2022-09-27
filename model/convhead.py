from tokenize import group
import torch
import torch.nn as nn
import math

class Head(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.clss = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,1)
        )

        self.reg = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,4)
        )
        # self.center = Center(dim)
    def forward(self,x):
        cls = self.clss(x)
        reg = self.reg(x).sigmoid()
#         reg = torch.cat([xy,wh],dim = -1)
        return cls,reg,0


class ClsHead(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.clss = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,1)
        )

    def forward(self,x):
        cls = self.clss(x)
        return cls

class ConvBlock(nn.Module):
    def __init__(self,channel,out_channel,act = True) -> None:
        super().__init__()
        self.conv11 = nn.Conv2d(channel,channel//2,kernel_size=1)
        self.conv = nn.Conv2d(channel//2,channel//2,kernel_size=3,padding=1)
        self.conv12 = nn.Conv2d(channel//2,out_channel)
        self.relu = nn.LeakyReLU() if act else nn.Identity()
    def forward(self,x):
        x = x + self.conv12(self.conv(self.conv11(x)))
        x = self.relu(x)
        return x
    
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))



class DoubleHead(nn.Module):
    def __init__(self,dim,conv_dim = 256) -> None:
        super().__init__()
        self.clss = nn.Sequential(
            nn.Linear(dim,384),
            nn.ReLU(),
            nn.Linear(384,384),
            nn.ReLU(),
            nn.Linear(384,1)
        )
        self.xy1 = conv(dim,conv_dim)
        self.xy2 = conv(conv_dim,conv_dim//2)
        self.xy3 = conv(conv_dim//2,conv_dim//4)
        self.xy4 = nn.Conv2d(conv_dim//4,2,kernel_size=1)

        self.wh1 = conv(dim,conv_dim)
        self.wh2 = conv(conv_dim,conv_dim//2)
        self.wh3 = conv(conv_dim//2,conv_dim//4)
        self.wh4 = nn.Conv2d(conv_dim//4,2,kernel_size=1)
    
    def forward(self,x):
        cls = self.clss(x)
        B,N,C = x.shape
        L = int(math.sqrt(N))
        x = x.transpose(-1,-2).reshape(B,C,L,L)
        xy = self.get_xy(x).flatten(2,3).transpose(-1,-2)
        wh = self.get_wh(x).flatten(2,3).transpose(-1,-2)
        reg = torch.cat([xy,wh],dim = -1)
        return cls,reg,0
    def get_wh(self,x):
        x = self.wh1(x)
        x = self.wh2(x)
        x = self.wh3(x)
        x = self.wh4(x)
        return x.sigmoid()
    def get_xy(self,x):
        x = self.xy1(x)
        x = self.xy2(x)
        x = self.xy3(x)
        x = self.xy4(x)
        return x.sigmoid()
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Centerhead(nn.Module):
    def __init__(self,dim,length = 11) -> None:
        super().__init__()
        self.clss = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,1)
        )
        self.xy1 = conv(dim,256)
        self.xy2 = conv(256,128)
        self.xy3 = conv(128,64,kernel_size=1,padding = 0)
        self.x = nn.Conv2d(64,length,kernel_size=1)
        self.y = nn.Conv2d(64,length,kernel_size=1)
        
        self.wh1 = conv(dim,256)
        self.wh2 = conv(256,128)
        self.wh3 = conv(128,64,kernel_size=1,padding = 0)
        self.w = nn.Conv2d(64,length,kernel_size=1)
        self.h = nn.Conv2d(64,length,kernel_size=1)
        
        self.scale = 1.
        self.length = length
        with torch.no_grad():
            self.line = torch.linspace(0, 1,self.length).cuda().unsqueeze(-1)#[Len,1]
            
    def forward(self,x):
        cls = self.clss(x)
        B,N,C = x.shape
        L = int(math.sqrt(N))
        x = x.transpose(-1,-2).reshape(B,C,L,L)
        x_score,y_score = self.get_xy(x)#[B,N,Len]
        w_score,h_score = self.get_wh(x)#[B,N,Len]
        x = x_score @ self.line
        y = y_score @ self.line
        w = w_score @ self.line
        h = h_score @ self.line
        reg = torch.cat([x,y,w,h],dim = -1)
        return cls,reg,0
    def get_wh(self,x):
        x = self.wh1(x)
        x = self.wh2(x)
        x = self.wh3(x)
        w = self.w(x).flatten(2,3).transpose(-1,-2) * self.scale
        h = self.h(x).flatten(2,3).transpose(-1,-2) * self.scale
        return w.softmax(dim = -1),h.softmax(dim = -1)
    def get_xy(self,x):
        x = self.xy1(x)
        x = self.xy2(x)
        x = self.xy3(x)
        y = self.y(x).flatten(2,3).transpose(-1,-2) * self.scale
        x = self.x(x).flatten(2,3).transpose(-1,-2) * self.scale
        return x.softmax(dim = -1),y.softmax(dim = -1)