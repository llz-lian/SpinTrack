
import numpy as np
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
def drop_zeros(x):
    B,_,C = x.shape
    ba = torch.linspace(0,B-1,B).long()
    nonZeroRows = torch.abs(x).sum(dim=2) > 0
    x = x[nonZeroRows]#.reshape(B,-1,C)
    x = x.reshape(B,-1,C)
    return x,nonZeroRows

def drop_zeros_keep(x):
    B,_,C = x.shape
    ret = []
    idx = []
    for i in range(B):
        x_i,idx_i = drop_zeros(x[i].unsqueeze(0))
        ret.append(x_i)
        idx.append(idx_i)
    return ret,idx

def zero_pos(x):
    B,_,C = x.shape
    ZeroRows = torch.abs(x).sum(dim=2) == 0
    return ZeroRows
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.target_len - 1) * (2 * self.target_len - 1), num_heads))
        self.relative_position_bias_table_temp = nn.Parameter(
            torch.zeros((2 * self.temp_len - 1) * (2 * self.temp_len - 1), num_heads)
        )
        self.pos_bias = 0

        if self.use_bias:
            temp_table = torch.from_numpy(gaussian_heatmap((temp_len//2,temp_len//2),(temp_len,temp_len))).float().flatten().cuda() \
                + torch.linspace(-0.2,0.2,temp_len**2,dtype=torch.float32).cuda()
            targ_table = torch.from_numpy(gaussian_heatmap((target_len//2,target_len//2),(target_len,target_len))).float().flatten().cuda() \
                + torch.linspace(-0.2,0.2,target_len**2,dtype=torch.float32).cuda()
            #[1,temp,target,2]
            self.temp_target_bias_table = torch.stack(torch.meshgrid([temp_table,targ_table])).permute(1,2,0).contiguous()
            #[1,target,temp,2]
            self.target_temp_bias_table = self.temp_target_bias_table.transpose(0,1)
            self.pos_mlp = nn.Sequential(
                nn.Linear(2,512,True),
                nn.ReLU(True),
                nn.Linear(512,num_heads)
            )

            nn.init.trunc_normal_(self.relative_position_bias_table, std=.0002)
            nn.init.trunc_normal_(self.relative_position_bias_table_temp, std=.0002)

            self.temp_bias_index = self.getBiasIndex(temp_len,temp_len)
            self.target_bias_index = self.getBiasIndex(target_len,target_len)
            self.pos_bias = None
    def getBiasIndex(self,w_len,h_len):
        coords_h = torch.arange(h_len)
        coords_w = torch.arange(w_len)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += w_len - 1  # shift to start from 0
        relative_coords[:, :, 1] += h_len - 1
        relative_coords[:, :, 0] *= 2 * w_len - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        return relative_position_index

    def track_init(self):
        temp_relative_position_bias = self.relative_position_bias_table_temp[self.temp_bias_index.view(-1)].view(
                self.temp_len * self.temp_len, self.temp_len * self.temp_len, -1)#[64,64,8]
        target_relative_position_bias = self.relative_position_bias_table[self.target_bias_index.view(-1)].view(
            self.target_len * self.target_len, self.target_len * self.target_len, -1)#[256,256,8]
        temp_relative_position_bias = temp_relative_position_bias.permute(2, 0, 1).contiguous()#[num_heads,temp,temp]
        target_relative_position_bias = target_relative_position_bias.permute(2, 0, 1).contiguous()#[num_heads,targ,targ]
        #[temp,target] [target,temp]
        target_temp_bias = self.pos_mlp(self.target_temp_bias_table).permute(2,0,1).contiguous()#[target,temp,num_heads]
        temp_target_bias = target_temp_bias.transpose(-1,-2)#[temp,target,num_heads]
        relative_position_bias =torch.cat((torch.cat((temp_relative_position_bias,temp_target_bias),dim = 2),
                                        torch.cat((target_temp_bias,target_relative_position_bias),dim = 2)),
                                        dim = 1).unsqueeze(0)
        self.pos_bias = relative_position_bias


    def forward(self,x,temp_q,target_q):
        B, N, C = x.shape
        #[B,nt+ns,dim] split
        #q:[B,ns,dim] kv:[B,nt+ns,dim*2]
        B,Ns,C = temp_q.shape

        q = torch.cat([temp_q,target_q],dim = 1)
        q = self.q(q).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        k = self.k(x).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        v = self.v(x).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        

        relative_position_bias = 0.
        if self.use_bias and self.pos_bias is None:
            temp_relative_position_bias = self.relative_position_bias_table_temp[self.temp_bias_index.view(-1)].view(
                self.temp_len * self.temp_len, self.temp_len * self.temp_len, -1)#[64,64,8]
            target_relative_position_bias = self.relative_position_bias_table[self.target_bias_index.view(-1)].view(
                self.target_len * self.target_len, self.target_len * self.target_len, -1)#[256,256,8]

            temp_relative_position_bias = temp_relative_position_bias.permute(2, 0, 1).contiguous()#[num_heads,temp,temp]
            target_relative_position_bias = target_relative_position_bias.permute(2, 0, 1).contiguous()#[num_heads,targ,targ]

            #[temp,target] [target,temp]
            target_temp_bias = self.pos_mlp(self.target_temp_bias_table).permute(2,0,1).contiguous()#[target,temp,num_heads]
            temp_target_bias = target_temp_bias.transpose(-1,-2)#[temp,target,num_heads]

            relative_position_bias =torch.cat((torch.cat((temp_relative_position_bias,temp_target_bias),dim = 2),
                                        torch.cat((target_temp_bias,target_relative_position_bias),dim = 2)),
                                        dim = 1).unsqueeze(0)
        else:
            relative_position_bias = self.pos_bias
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
                 use_conv_bias = True) -> None:
        super().__init__()
        self.temp_num = temp_len**2
        self.temp_window_size = temp_len//2
        self.target_window_size = target_len//2
        self.temp_len = temp_len
        self.targ_len =target_len
        self.attn = Convattn(dim,num_heads,qkv_bias,qk_scale,attn_drop_ratio,proj_drop_ratio,temp_len//2,target_len//2,relative_bias)

        #1,3:down 2,4:right 0:nothing
        self.roll_func = self.getRoll(roll_type)
        # if target_len == 16:
        #     temp_len = 12
        #     target_len = 24
        # if target_len == 8:
        #     temp_len = 6
        #     target_len = 12
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
        self.token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.token, mean=0., std=.02)
    def track_init(self):
        self.attn.track_init()

    def forward(self,x,index = None,ba = None):
        temp,target = x[:,0:self.temp_num,:],x[:,self.temp_num:,:]
        target[ba,index,:] = self.token
        B,_,C = temp.shape
        #[B,N,C] => [B,H,W,C]
        temp = temp.reshape(B,self.temp_len,self.temp_len,C)
        target = target.reshape(B,self.targ_len,self.targ_len,C)
        #pos
        temp_ = temp
        target_ = target
        if self.temppos is not None:
            temp_ = temp + self.temppos(temp.transpose(1,3)).transpose(1,3)
            target_ = target + self.targpos(target.transpose(1,3)).transpose(1,3)
        temp_roll = self.roll_func(temp_,self.temp_window_size)
        target_roll = self.roll_func(target_,self.target_window_size)
        temp_roll = window_partition(temp_roll,self.temp_window_size).flatten(1,2)
        target_roll = window_partition(target_roll,self.target_window_size).flatten(1,2)
        #[nW*B,win*win,dim]
        temp_win = window_partition(temp,self.temp_window_size).flatten(1,2)
        targ_win = window_partition(target,self.target_window_size).flatten(1,2)
        B,temp_len,C = temp_win.shape
        x = torch.cat([temp_win,targ_win],dim = 1)
        x = self.attn(x,temp_roll,target_roll)
        temp = x[:,0:temp_len,:]
        target = x[:,temp_len:,:]
        temp = window_reverse(temp,self.temp_window_size,self.temp_len,self.temp_len).flatten(1,2)
        target = window_reverse(target,self.target_window_size,self.targ_len,self.targ_len).flatten(1,2)
        return torch.cat([temp,target],dim = 1)

    def _roll_right(self,x,shift_size):
        #x:[B,H,W,C]
        shifted_x = torch.roll(x, shifts=(0, -shift_size), dims=(1, 2))
        return shifted_x
    def _roll_down(self,x,shift_size):
        shifted_x = torch.roll(x, shifts=(-shift_size,0), dims=(1, 2))
        return shifted_x
    def _roll_down_right(self,x,shift_size):
        shifted_x = torch.roll(x, shifts=(-shift_size,-shift_size), dims=(1, 2))
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
                 use_conv_bias = True
                 ):
        super(swinCrossBasic, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = swinCross(dim = dim,num_heads=num_heads,qkv_bias=qkv_bias,relative_bias=relative_bias
        ,qk_scale=qk_scale,temp_len=temp_len,target_len=target_len,roll_type=roll_type,useV2=useV2,use_conv_bias=use_conv_bias)
         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio) 
        self.use_checkpoint = use_checkpoint
        self.temp_token_num = temp_len**2
        self.need = torch.ones([1,target_len**2],dtype=bool).to(device)
    def track_init(self):
        self.attn.track_init()
    def forward(self, x,index = None):

        if self.use_checkpoint:
            x = cp.checkpoint(self._forward_nc,x,index)
        else:
            x = self._forward_nc(x,index)
        return x
    def _forward_nc(self,x,index = None):
        B,_,C = x.shape
        x = self.norm1(x)
        ba = torch.linspace(0,B-1,B).long().unsqueeze(-1)
        x = x + self.drop_path(self.attn(x,index,ba))
        #mlp
        temp = x[:,0:self.temp_token_num,:]
        temp = temp + self.drop_path(self.mlp(self.norm2(temp)))
        targ = x[:,self.temp_token_num:,:]#.reshape(-1,C)
        need = self.need.clone().repeat(B,1)
        pad = targ.clone()
        need[ba,index] = 0
        # pad = pad.reshape(-1,C)
        # nonZeroRows = torch.abs(pad).sum(dim=-1) > 0
        targ_need = targ[need]
        targ_need =targ_need + self.drop_path(self.mlp(self.norm2(targ_need)))

        pad[need] = targ_need
        #pad = pad.reshape(B,-1,C)
        x = torch.cat([temp,pad],dim = 1)
        return x

class CrossBlock(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio = 4) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim,num_heads,batch_first=True)
        self.mlp = Mlp(dim,dim*mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self,features,target):
        B,_,C = features.shape
        features,_ = drop_zeros(features)
        features = self.norm1(features)
        target = self.norm1(target)
        target = target +self.cross(target,features,features)[0]
        target = target + self.mlp(self.norm2(target))
        return target

class ForwardBlockDrop(nn.Module):
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
                drop_targ = False,
                use_conv_bias = True) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(base_ratio,drop_ratio, depth)]
        self.spin_block = nn.ModuleList()
        for i in range(depth):
            layer = swinCrossBasic(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,qk_scale=None,relative_bias = relative_bias,
                  drop_ratio=drop_ratio, drop_path_ratio=dpr[i],temp_len = temp_len,target_len = target_len,useV2 = useV2,
                  norm_layer=nn.LayerNorm, act_layer=nn.GELU,roll_type= i % 4,use_checkpoint=use_checkpoint,use_conv_bias=use_conv_bias)
            self.spin_block.append(layer)
       
        self.swin_block_layers = nn.ModuleList()
        self.down_sample = PatchMerging(embed_dim) if change_dim else None
        self.conv_increase =nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim,int(embed_dim*1.5),kernel_size=3,padding=1,groups=embed_dim//num_heads)
        ) if change_dim else None

        for i_layer in range(2):
            layer = BasicLayer(dim= embed_dim,
            depth= 2,num_heads=num_heads,window_size= window_size,mlp_ratio= mlp_ratio,qkv_bias= qkv_bias,
            drop_path= dpr[i_layer+1],use_checkpoint=use_checkpoint,input_resolution=(swin_tlen,swin_tlen),temp_resolution=(swin_tlen,swin_tlen))
            self.swin_block_layers.append(layer)

        self.cross = CrossBlock(embed_dim,num_heads,mlp_ratio)
        self.cross2 = CrossBlock(embed_dim,num_heads,mlp_ratio) if two_cross else None

        self.drop = DropPath(drop_ratio)
        self.hw = target_len
        self.temphw = temp_len
        self.swintlen = swin_tlen
        self.drop_targ = drop_targ
    def track_init(self):
        for layer in self.spin_block:
            layer.track_init()
    def forward(self,features,target):
        B,_,C = target.shape
        index = None
        index,w = self.getDrop(features)

        for layer in self.spin_block:
            features = layer(features,index)

        target = self.swin_block_layers[0](target)
        if self.cross2 is not None:
            target = self.cross2(features,target)
        target = self.swin_block_layers[1](target)
        target = self.cross(features,target)

        if self.down_sample is not None:
            temp = features[:,0:self.temphw**2,:]
            targ = features[:,self.temphw**2:,:]
    
            temp = self.down_sample(temp,self.temphw,self.temphw)
            targ = self.down_sample(targ,self.hw,self.hw)
            features = torch.cat([temp,targ],dim = 1)
            target = target.transpose(-1,-2).reshape(B,C,self.swintlen,self.swintlen)
            target = self.conv_increase(target).flatten(2,3).transpose(-1,-2)

        return features,target
    def getDrop(self,features):
        temp = features[:,0:self.temphw**2,:]
        targ = features[:,self.temphw**2:,:]
        idx = self.temphw//2 +self.temphw * self.temphw//2 - 1
        w = F.normalize(temp[:,idx:(idx+1),:],dim = -1) @ F.normalize(targ,dim = -1).transpose(-1,-2)
        drop_rate = 0.3
        drop_num = int(drop_rate*(self.hw**2))
        values,index = w.topk(drop_num,dim = -1,largest=False)

        index = index.squeeze(1)#.transpose(-1,-2)
        return index,w
    def getDropIndex(self,weight,drop_num,bias):
        #bias[x,y]
        values,index = weight.topk(drop_num,dim = -1,largest=False)


if __name__ == '__main__':
    temp = torch.cat([torch.ones([1,128,384]),torch.zeros([1,128,384])],dim = 1)
    target = torch.zeros([1,576,384])
    x = torch.cat([temp,target],dim = 1)
    model = swinCrossBasic(dim=384,num_heads=16,temp_len=16,target_len=24)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    out = model(x)
    # print(out.shape)