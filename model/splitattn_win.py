
import numpy as np
import math
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from .swin import SwinTransformer, window_partition,window_reverse,DropPath,Mlp,BasicLayer
def gaussian_heatmap(center = (2, 2), image_size = (10, 10), sig = 1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, int(1.5 * dim), bias=False)
        self.norm = norm_layer(4 * dim)
        # self.norm = nn.BatchNorm2d(dim)
    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

class Convattn(nn.Module):
    def __init__(self, 
                 dim,   # 输入token的dim
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 temp_len = 16,# 图像边长
                 target_len = 22,
                 relative_bias = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5        
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.q = nn.Linear(dim,dim,qkv_bias)
        self.k = nn.Linear(dim,dim,qkv_bias)
        self.v = nn.Linear(dim,dim,qkv_bias)       


        self.proj = nn.Linear(dim,dim)
        
        self.use_bias = relative_bias
        self.temp_len = temp_len
        self.target_len = target_len
        self.num_token = self.temp_len**2

        self.pos_bias = 0
#         print('attn:{}'.format(self.use_bias))
        if self.use_bias:
            self.pos_bias = None

    def track_init(self):
        pass


    def forward(self,x,temp_q,target_q,bias):
        B, N, C = x.shape
        #[B,nt+ns,dim] split
        #q:[B,ns,dim] kv:[B,nt+ns,dim*2]
        B,Ns,C = temp_q.shape

        q = torch.cat([temp_q,target_q],dim = 1)
        q = self.q(q).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        k = self.k(x).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        v = self.v(x).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        
        #[B,N,N,heads]
        relative_position_bias = 0
        if self.use_bias:
            relative_position_bias = bias.permute(0,3,1,2)
        #[B,heads,Ns,dim_heads] * [B,heads,dim_per_head,Nt+ns] ==> [B,heads,Ns,Nt+ns]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + relative_position_bias
        attn = attn.softmax(dim=-1) #[B,heads,num_patch,num_patch]
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class RelTabel(nn.Module):
    def __init__(self,dim,temp_len,target_len,num_heads):
        super().__init__()
        self.dim = dim
        self.temp_len = temp_len
        self.target_len = target_len
        self.num_heads = num_heads

        self.pos_mlp = nn.Sequential(
                nn.Linear(2,512,True),
                nn.ReLU(True),
                nn.Linear(512,num_heads)
            )
        self.temp_table = torch.from_numpy(gaussian_heatmap(((temp_len-1)/2,(temp_len-1)/2),(temp_len,temp_len),sig = temp_len//4)).float().flatten() \
                + torch.linspace(-0.2,0.2,temp_len**2,dtype=torch.float32)
        self.targ_table = torch.from_numpy(gaussian_heatmap(((target_len-1)/2,(target_len-1)/2),(target_len,target_len),sig = target_len//4)).float().flatten() \
                + torch.linspace(-0.2,0.2,target_len**2,dtype=torch.float32)
        #[1,H,W,1]
        self.temp_table = self.temp_table.reshape(1,temp_len,temp_len,1).cuda()
        self.targ_table = self.targ_table.reshape(1,target_len,target_len,1).cuda()
    def forward(self,roll_func):
        temp_win = self.temp_len//2
        targ_win = self.target_len//2
        temp_rel_q = roll_func(self.temp_table,temp_win)
        targ_rel_q = roll_func(self.targ_table,targ_win)
        #[4,win*win]
        temp_rel_q = window_partition(temp_rel_q,temp_win).flatten(1)
        targ_rel_q = window_partition(targ_rel_q,targ_win).flatten(1)
        temp_rel = window_partition(self.temp_table,temp_win).flatten(1)
        targ_rel = window_partition(self.targ_table,targ_win).flatten(1)

        b0 = self.get_bias(temp_rel_q[0],targ_rel_q[0],temp_rel[0],targ_rel[0])
        b1 = self.get_bias(temp_rel_q[1],targ_rel_q[1],temp_rel[1],targ_rel[1])
        b2 = self.get_bias(temp_rel_q[2],targ_rel_q[2],temp_rel[2],targ_rel[2])
        b3 = self.get_bias(temp_rel_q[3],targ_rel_q[3],temp_rel[3],targ_rel[3])
        bias = torch.cat([b0,b1,b2,b3],dim = 0)
        return bias
    def get_bias(self,temp_rel_q,targ_rel_q,temp_rel,targ_rel):
        #[TT][1,temp,temp,2]
        tt = torch.stack(torch.meshgrid([temp_rel_q,temp_rel])).permute(1,2,0).contiguous().unsqueeze(0)
        #[TS][1,temp,targ,2]
        ts = torch.stack(torch.meshgrid([temp_rel_q,targ_rel])).permute(1,2,0).contiguous().unsqueeze(0)
        #[ST][1,targ,temp,2]
        st = torch.stack(torch.meshgrid([targ_rel_q,temp_rel])).permute(1,2,0).contiguous().unsqueeze(0)
        #[SS][1,targ,targ,2]
        ss = torch.stack(torch.meshgrid([targ_rel_q,targ_rel])).permute(1,2,0).contiguous().unsqueeze(0)
        tt = self.pos_mlp(tt)
        ts = self.pos_mlp(ts)
        st = self.pos_mlp(st)
        ss = self.pos_mlp(ss)
        bias = torch.cat([
                torch.cat([tt,ts],dim = 2),
                torch.cat([st,ss],dim = 2)
               ],dim = 1)
        return bias




class swinCross(nn.Module):
    def __init__(self, 
                 dim,   # 输入token的dim
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 temp_len = 16,# 图像边长
                 target_len = 22,
                 roll_type = 1,
                 useV2 = False,
                 relative_bias = True,
                 use_conv_bias = True,
                 roll_kv = False) -> None:
        super().__init__()
        self.roll_kv = roll_kv
        self.temp_num = temp_len**2
        self.temp_window_size = temp_len//2
        self.target_window_size = target_len//2
        self.temp_len = temp_len
        self.targ_len =target_len
        self.attn = Convattn(dim,num_heads,qkv_bias,qk_scale,attn_drop_ratio,proj_drop_ratio,temp_len//2,target_len//2,relative_bias = relative_bias)
#         print('swinCross:{}'.format(relative_bias))
        self.rel_bias = None
        if relative_bias:
            self.rel_bias = RelTabel(dim,temp_len,target_len,num_heads)
        #1,3:down 2,4:right 0:nothing
        self.roll_func = self.getRoll(roll_type)
        print('roll type:{}'.format(roll_type))
        if target_len == 16:
            temp_len = 12
            target_len = 24
        if target_len == 8:
            temp_len = 6
            target_len = 12
        temp_k = temp_len//2 -1
        targ_k = target_len//2 -1
        if temp_k %2 ==0 :
            temp_k += 1
        if targ_k %2 ==0:
            targ_k += 1
        self.temppos = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim,dim,kernel_size=temp_k,padding=(temp_k-1)//2,groups=dim,bias=False)
        ) if use_conv_bias else None
        self.targpos = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim,dim,kernel_size=targ_k,padding=(targ_k-1)//2,groups=dim,bias = False)
        ) if use_conv_bias else None
        self.bias = None
    def track_init(self):
        self.bias = self.rel_bias(self.roll_func)

    def forward(self,x):
        temp,target = x[:,0:self.temp_num,:],x[:,self.temp_num:,:]
        B,_,C = temp.shape

        #bias
        bias = self.bias
        if self.rel_bias is not None and self.bias is None:
            bias = self.rel_bias(self.roll_func)
            bias = bias.repeat(B,1,1,1)
        #[B,N,C] => [B,H,W,C]
        temp = temp.reshape(B,self.temp_len,self.temp_len,C)
        target = target.reshape(B,self.targ_len,self.targ_len,C)

        #pos
        if self.temppos is not None:
            temp = temp + self.temppos(temp.transpose(1,3)).transpose(1,3)
            target = target + self.targpos(target.transpose(1,3)).transpose(1,3)

        temp_roll = self.roll_func(temp,self.temp_window_size)
        target_roll = self.roll_func(target,self.target_window_size)
        temp_roll = window_partition(temp_roll,self.temp_window_size).flatten(1,2)
        target_roll = window_partition(target_roll,self.target_window_size).flatten(1,2)

        #[nW*B,win*win,dim]
        temp_win = window_partition(temp,self.temp_window_size).flatten(1,2)
        targ_win = window_partition(target,self.target_window_size).flatten(1,2)
        B,temp_len,C = temp_win.shape
        x = torch.cat([temp_win,targ_win],dim = 1)
        if self.roll_kv:
            x = self.attn(torch.cat([temp_roll,target_roll],dim = 1),temp_win,targ_win,bias)
        else:
            x = self.attn(x,temp_roll,target_roll,bias)

        temp = x[:,0:temp_len,:]
        target = x[:,temp_len:,:]
        temp = window_reverse(temp,self.temp_window_size,self.temp_len,self.temp_len).flatten(1,2)
        target = window_reverse(target,self.target_window_size,self.targ_len,self.targ_len).flatten(1,2)

        return torch.cat([temp,target],dim = 1)

    def _roll_right(self,x,shift_size):
        #x:[B,H,W,C]
        # shifted_x = torch.roll(x, shifts=(0, -shift_size), dims=(1, 2))
        shifted_x = torch.rot90(x,3,dims=(1,2))
        return shifted_x
    def _roll_down(self,x,shift_size):
        # shifted_x = torch.roll(x, shifts=(-shift_size,0), dims=(1, 2))
        shifted_x = torch.rot90(x,1,dims=(1,2))
        return shifted_x
    def _roll_down_right(self,x,shift_size):
        # shifted_x = torch.roll(x, shifts=(-shift_size,-shift_size), dims=(1, 2))
        shifted_x = torch.rot90(x,2,dims=(1,2))
        return shifted_x
    def _roll_nothing(self,x,shift_size):

        return x

    def getRoll(self,type):
        if type == 0:
            return self._roll_nothing
        if type == 1:
            return self._roll_down
        if type == 2:
            return self._roll_right
        if type == 3:
            return self._roll_down_right
        return None

class swinCrossBasic(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 relative_bias = True,
                 useV2 = True,
                 qk_scale=None,
                 drop_ratio=0.,
                 drop_path_ratio=0.,
                 temp_len = 16,
                 target_len = 22,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 roll_type = 0,
                 use_checkpoint = True,
                 few_ffn = False,
                 use_conv_bias = True,
                 roll_kv = False
                 ):
        super(swinCrossBasic, self).__init__()
#         print('swinCrossBasic:{}'.format(relative_bias))

        self.norm1 = norm_layer(dim)
        self.attn = swinCross(dim = dim,num_heads=num_heads,qkv_bias=qkv_bias,relative_bias=relative_bias
        ,qk_scale=qk_scale,temp_len=temp_len,target_len=target_len,roll_type=roll_type,useV2=useV2,use_conv_bias=use_conv_bias,roll_kv = roll_kv)
         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if not few_ffn or (few_ffn and roll_type == 3) :
            # print('rolltype:{},ffn'.format(roll_type))
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio) 
        else:
            self.mlp = None
        self.use_checkpoint = use_checkpoint

    def track_init(self):
        self.attn.track_init()
    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # return x
        if self.use_checkpoint:
            x = cp.checkpoint(self._forward_nc,x)
        else:
            x = self._forward_nc(x)
        return x
    def _forward_nc(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossBlock(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio = 4,temp_len = 12) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim,num_heads,batch_first=True)
        self.mlp = Mlp(dim,dim*mlp_ratio)
        self.not_all = False
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def set_not_all(self):
        self.not_all = True
    def forward(self,features,target,ba,target_index = None):
        features = self.norm1(features)
        if target_index is not None:
            foreground = target[ba,target_index].type_as(target)
            foreground = self.norm1(foreground)
            foreground = self.cross(foreground,features,features)[0]
            target[ba,target_index] = target[ba,target_index] + foreground
            target[ba,target_index] = target[ba,target_index] + self.mlp(self.norm2(target[ba,target_index]))
        else:
            target = self.norm1(target)
            target = target +self.cross(target,features,features)[0]
            target = target + self.mlp(self.norm2(target))
        return target,None,None


class ForwardBlock(nn.Module):
    def __init__(self,
                embed_dim = 384,
                num_heads = 8,
                mlp_ratio = 4,
                base_ratio = 0,
                drop_ratio = 0.2,
                window_size = 7,
                depth = 4,
                temp_len = 16,
                target_len = 24,
                swin_tlen = 24,
                qkv_bias = True,
                relative_bias = True,
                useV2 = False,
                use_checkpoint = True,
                change_dim = True,
                two_cross = True,
                few_ffn = False,
                use_conv_bias = True,use_roll = True,roll_kv = False) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(base_ratio,drop_ratio, depth)]
        self.spin_block = nn.Sequential(*[
            swinCrossBasic(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_scale=None,relative_bias = relative_bias,
                  drop_ratio=drop_ratio, drop_path_ratio=dpr[i],temp_len = temp_len,target_len = target_len,useV2 = useV2,few_ffn = few_ffn,
                  norm_layer=nn.LayerNorm, act_layer=nn.GELU,roll_type= i % 4 if use_roll else 0,use_checkpoint=use_checkpoint,use_conv_bias=use_conv_bias,roll_kv = roll_kv)
            for i in range(depth)
            ])
        self.swin_block_layers = nn.ModuleList()
        self.down_sample = PatchMerging(embed_dim) if change_dim else None

        self.conv_increase = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim,int(1.5*embed_dim),kernel_size=3,padding=1,groups=embed_dim//num_heads)
        ) if change_dim else None

        for i_layer in range(2):
            layer = BasicLayer(dim= embed_dim,
            depth= 2,num_heads=num_heads,window_size= window_size,mlp_ratio= mlp_ratio,qkv_bias= qkv_bias,
            drop_path= dpr[i_layer+1],use_checkpoint=False,input_resolution=(swin_tlen,swin_tlen),temp_resolution=(swin_tlen,swin_tlen))
            self.swin_block_layers.append(layer)

        self.cross = CrossBlock(embed_dim,num_heads,mlp_ratio,temp_len)
        self.cross2 = CrossBlock(embed_dim,num_heads,mlp_ratio,temp_len) if two_cross else None

        self.hw = target_len
        self.temphw = temp_len
        self.swintlen = swin_tlen
        self.use_checkpoint = use_checkpoint
    def track_init(self):
        for layer in self.spin_block:
            layer.track_init()
    def downSample(self,features,target):
        if self.down_sample is not None:
            temp = features[:,0:self.temphw**2,:]
            targ = features[:,self.temphw**2:,:]
            # target = torch.cat([target,targ],dim = -1)
            temp = self.down_sample(temp,self.temphw,self.temphw)
            targ = self.down_sample(targ,self.hw,self.hw)
            features = torch.cat([temp,targ],dim = 1)
            B,_,C = target.shape
            target = target.transpose(-1,-2).reshape(B,C,self.swintlen,self.swintlen)
            target = self.conv_increase(target)#.flatten(2,3).transpose(-1,-2)
            target = target.flatten(2,3).transpose(-1,-2)
        return features,target

    def forward_cross(self,features,target,decoder_target_index = None,encoder_target_index = None):
        temp = features[:,0:self.temphw**2,:]
        targ = features[:,self.temphw**2:,:]
        B = temp.shape[0]
        ba = torch.linspace(0,B-1,B).long().unsqueeze(-1)
        if encoder_target_index is not None:
            targ = targ[ba,encoder_target_index]

        features = torch.cat([temp,targ],dim = 1)
        target,w,drop_index = self.cross(features,target,ba,decoder_target_index)
        target = self.swin_block_layers[1](target)

        return target,w,drop_index

    def forward(self,features,target):
        features = self.spin_block(features)
        target = self.swin_block_layers[0](target)
#         target = self.swin_block_layers[1](target)
        # target,w,drop_index = self.cross(features,target)

        return features,target#,w,drop_index

    def set_not_all(self):
        pass

if __name__ == '__main__':
    temp = torch.cat([torch.ones([1,128,384]),torch.zeros([1,128,384])],dim = 1)
    target = torch.zeros([1,576,384])
    x = torch.cat([temp,target],dim = 1)
    model = swinCrossBasic(dim=384,num_heads=16,temp_len=16,target_len=24)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    out = model(x)
    # print(out.shape)