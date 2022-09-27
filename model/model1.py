from turtle import forward
from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
import math
import torch.nn.functional as F


class RealInvo(nn.Module):
    def __init__(self,
                channels: int,
                sigma_mapping: Optional[nn.Module] = None,
                kernel_size: int = 7,
                stride: Union[int, Tuple[int, int]] = (1, 1),
                groups: int = 1,
                reduce_ratio: float = 1,
                dilation: Union[int, Tuple[int, int]] = (1, 1),
                padding: Union[int, Tuple[int, int]] = (3, 3),
                bias: bool = False,
                **kwargs):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.ratio = reduce_ratio
        self.dilation = dilation #?
        self.padding = padding
        self.bias = bias

        self.tnum = math.ceil( float(channels)/reduce_ratio)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels,self.tnum,kernel_size=1,stride=stride),
            nn.BatchNorm2d(self.tnum,eps=1e-6),
            nn.ReLU(inplace=True)
        )

        self.span = nn.Sequential(
            nn.Conv2d(self.tnum,self.groups * self.kernel_size**2,kernel_size=1)
        )
        self.avg = nn.AvgPool2d(kernel_size=2,stride=1)

        self.unfold = nn.Unfold(kernel_size=kernel_size,stride=stride,padding=(kernel_size-1)//2)
    def forward(self,input:torch.Tensor):
        batch_size,_,h,w = input.shape

        #init kernel
        kernel = self.span(self.reduce(input))#b,h*w*1 * groups * k * k
        kernel = kernel.view(batch_size,self.groups,self.kernel_size**2,h,w).unsqueeze(2)#b,groups,1,k,k,h,w

        #prepare input
        output = self.unfold(input)#b,k,k,(h-),(w-)
        output = output.view(batch_size,self.groups,self.channels//self.groups,self.kernel_size**2,h,w)#b,groups,c_per_g,k,k,h,w

        output = (output * kernel).sum(dim=3)

        return output.view(batch_size,_,h,w)




class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Embed(nn.Module):
    def __init__(self,img_size = 256,embed_dim = 512,in_channel = 3,patch_size = 28):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.porj = nn.Conv2d(in_channel,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        """
        x:[B,C,H,W]
        output:[B,HW,C]
        """

        x = self.porj(x).flatten(2).transpose(1,2)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
     Basic block
    """
    def __init__(self,in_features = 512,num_heads = 8,qkv_bias = True,mlp_ratio = 2,drop = 0.):
        super().__init__()
        #self.attn = nn.MultiheadAttention(in_features,num_heads=num_heads)
        self.attn = Attention(dim=in_features,num_heads=num_heads,qkv_bias=qkv_bias)
        hidden = int(in_features * mlp_ratio)
        self.mlp = Mlp(in_features, hidden_features= hidden,drop= drop) if mlp_ratio>0 else nn.Identity()
        self.norm1 = nn.LayerNorm(in_features,eps=1e-6)
        self.norm2 = nn.LayerNorm(in_features,eps=1e-6)

    def forward(self,features):
        features = features + self.attn(self.norm1(features))
        features = features + self.mlp(self.norm2(features))
        return features

class Regression(nn.Module):#useless
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeatureBlock(nn.Module):
    def __init__(self,in_channel = 3,mid_channel = 3,out_channel = 3,
    #ks1 = 7,pd1 = 3,ks2 = 5,pd2 = 2,ks3 = 3,pd3 = 1,groups = 3
    ks1 = 1,pd1 = 0,
    ks2 = 3,pd2 = 1,
    ks3 = 1,pd3 = 0,groups = 1
    ) -> None:
        super().__init__()
        self.invo1 = nn.Sequential(
            RealInvo(mid_channel,groups=groups),
            RealInvo(mid_channel,groups=groups),
            nn.BatchNorm2d(mid_channel,eps=1e-6),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,mid_channel,kernel_size=ks1,padding=pd1),
            nn.Conv2d(mid_channel,mid_channel,kernel_size=ks2,padding=pd2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel,mid_channel,kernel_size=ks2,padding=pd2),
            nn.Conv2d(mid_channel,out_channel,kernel_size=ks1,padding=pd1)
        )
        self.conv3 = nn.Identity() if in_channel == out_channel else nn.Conv2d(in_channel,out_channel,kernel_size=ks3,padding=pd3)

    def forward(self,x):
        out = self.conv3(x)
        x = self.conv1(x)
        x = self.invo1(x)
        x = self.conv2(x)
        out = out + x
        return out




class Imgtotoken(nn.Module):
    def __init__(self,in_channel = 3,mid_rates = [1,2,1,1,2],out_channel = 32):
        super().__init__()
        self.in_channel = 3
        self.mid_rates = mid_rates
        self.out_channel = out_channel
        self.blocks = self.buildBlocks()
    def forward(self,x):
        out = self.blocks(x)
        return out
    
    def buildBlocks(self):
        blocks = []
        channel = self.in_channel
        for mid_rate in self.mid_rates:
            mid_channel = mid_rate * channel
            blocks.append(FeatureBlock(channel,mid_channel,mid_channel))
            channel = mid_channel
        blocks.append(FeatureBlock(channel,channel,self.out_channel))
        return nn.Sequential(*blocks)


class FeatureBlock2(nn.Module):
    def __init__(self,in_channel,mid_rate,out_channel) -> None:
        super().__init__()
        mid_channel = int(mid_rate*out_channel)
        self.invo = RealInvo(mid_channel,groups=mid_channel)
        self.conv1 = nn.Conv2d(in_channel,mid_channel,kernel_size = 7,padding = 3,groups=in_channel)

        self.conv2 = nn.Conv2d(mid_channel,in_channel,kernel_size= 1) if in_channel == out_channel else nn.Conv2d(mid_channel,out_channel,kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(mid_channel,eps=1e-6)

        
        self.conv3 = nn.Identity() if in_channel == out_channel else nn.Conv2d(in_channel,out_channel,kernel_size=1)
       
    def forward(self,x):
        input = self.conv3(x)
        x = self.conv1(x)
        x = self.invo(x)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C) 
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.act(x)

        x = self.conv2(x)
        x = input + x
        return x


class Imgtotoken2(nn.Module):
    def __init__(self,in_channel = 3,mid_rates = 2,out_channel = [3,3,3,6,6,12]) -> None:
        super().__init__()
        self.in_channel = 3
        self.mid_rates = mid_rates
        self.out_channel = out_channel
        self.blocks = self.buildBlocks()
    def buildBlocks(self):
        blocks = []
        channel = self.in_channel
        for out_channel in self.out_channel:
            blocks.append(FeatureBlock2(channel,self.mid_rates,out_channel))
            channel = out_channel
        return nn.Sequential(*blocks)
    def forward(self,x):
        return self.blocks(x)



class My(nn.Module):
    def __init__(self,
    target_size = 256,
    template_size = 128,
    in_channel = 3,
    embed_dim = 384,
    num_heads = 8,
    depth = 8,
    num_addon = 4,
    target_patch = 16,#16
    template_patch = 8,#7
    mlp_ratio = 2,
    drop = 0.1,
    num_classes = 1,
    divided = False,
    temp_mid_rates = [1,2,1,1,2],
    temp_out_channel = 32,
    mid_rates = 3):
        super().__init__()
        self.temp_embed = Embed(img_size=template_size,embed_dim=embed_dim,in_channel=12,patch_size=template_patch)
        self.temp_pre_embed = Imgtotoken2(in_channel,mid_rates)
        self.temp_blocs = nn.Sequential(*[Block(embed_dim,num_heads//4,mlp_ratio=1,drop = drop) for i in range(depth//4)])

        self.targ_embed = Embed(img_size=target_size,embed_dim=embed_dim,in_channel=in_channel,patch_size=target_patch)
        #self.targ_pre_embed = Imgtotoken2(in_channel=3,mid_rates=1,out_channel=[3])

        self.blocks = nn.Sequential(*[Block(embed_dim,num_heads,mlp_ratio=mlp_ratio,drop = drop) for i in range(depth)])

        self.token_grid_len = template_size//template_patch
        self.num_token = self.token_grid_len **2
        self.num_in = (self.token_grid_len * (self.token_grid_len+1))//2
        self.num_addon = num_addon
        self.num_patch = (target_size//target_patch)**2
        self.divided = divided

        self.pos = nn.Parameter(torch.Tensor(1,self.num_patch + self.num_token + num_addon,embed_dim))
        self.addon = nn.Parameter(torch.Tensor(1,num_addon,embed_dim))
        
        
        self.last_self = nn.Sequential(*[Block(embed_dim,num_heads=num_heads//4,mlp_ratio=1,drop=drop) for i in range(depth//4)])

        self.reg = Regression(embed_dim,embed_dim,2,3)
        self.regpos = Regression(embed_dim,embed_dim,4,3)

        self.norm = nn.LayerNorm(embed_dim,eps = 1e-6)
        self.act = nn.Sigmoid()
        self.feature = None
    

    def forward(self,target,tp):
        #target = self.targ_pre_embed(target)
        
        target = self.targ_embed(target)
        
        tp = self.temp_pre_embed(tp)
        tp = self.temp_embed(tp)#[B,HW,C] B,16,512
        tp = self.temp_blocs(tp)
        

        addon_tokens = self.addon.expand(target.shape[0],-1,-1)
        
        features = torch.cat((tp,target,addon_tokens),dim = 1)
        features += self.pos
        features = self.blocks(features)
        features = self.norm(features)


        """
        tokens = features[:,0:self.num_token,:]#[B,16,512]
        addon_tokens = features[:,-self.num_addon:,:]
        
        features = torch.cat((tokens,addon_tokens),dim=1)
        """
        features = features[:,self.num_token:self.num_token+self.num_patch,:]#[B,16,512]
        
        features = self.last_self(features)

        box = self.act(self.regpos(features))
        classes = self.reg(features)
        output = {'pred_boxes':box,'pred_logits':classes}
        
        return output

    def template(self,tp):
        if self.feature is None:
            tp = self.temp_pre_embed(tp)
            tp = self.temp_embed(tp)#[B,HW,C] B,16,512
            tp = self.temp_blocs(tp)
            self.feature = tp
    
    def track(self,target):
        #self.template(tp)
        
        tp = self.feature
        
        addon_tokens = self.addon.expand(target.shape[0],-1,-1)
        


        target = self.targ_embed(target)
        features = torch.cat((tp,target,addon_tokens),dim = 1)
        features += self.pos
        features = self.blocks(features)
        features = self.norm(features)

        features = features[:,self.num_token:self.num_token+self.num_patch,:]#[B,16,512]
        
                
        box = self.act(self.regpos(features))
        classes = self.reg(features)
        output = {'pred_boxes':box,'pred_logits':classes}
        
        return output

from torchsummary import summary


if __name__ == "__main__":

    target = torch.randn(2,3,256,256)
    template = torch.randn(2,3,112,112)
    """
    classes,box = modle(target,template)
    classes = classes.squeeze(0)
    box = classes.squeeze(0)
    print(box)
    """