import imp
from typing import Union, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .swin_transformer import buildAll,buildBase,buildSmall,buildLarge
from .splitattn_win import swinCrossBasic,ForwardBlock,Convattn
from .splitattn_win_drop import ForwardBlockDrop
from .convhead import ConvHead,Head,ClassHead
from .swin_transformerv2 import getbaseV2
from .swin import getbase,getlarge,getsmall,getbase512,getlarge384
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)

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
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 temp_len = 16,# 图像边长
                 target_len = 22,
                 relative_bias = True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.use_bias = relative_bias
        #self.bias_len = temp_len if temp_len>target_len else target_len #选较大的就行
        
        self.temp_len = temp_len
        self.target_len = target_len
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.target_len - 1) * (2 * self.target_len - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.0002)

        self.relative_position_bias_table_temp = nn.Parameter(
            torch.zeros((2 * self.temp_len - 1) * (2 * self.temp_len - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table_temp, std=.0002)

        self.temp_target_bias_table = nn.Parameter(
            torch.zeros(num_heads,self.target_len**2,1)
        )
        self.target_temp_bias_table = nn.Parameter(
            torch.zeros(num_heads,self.temp_len**2,1)
        )
        self.temp_target_bias_line = nn.Parameter(
            torch.zeros(num_heads,1,self.temp_len**2)
        )
        self.target_temp_bias_line = nn.Parameter(
            torch.zeros(num_heads,1,self.target_len**2)
        )
        nn.init.trunc_normal_(self.temp_target_bias_line,  std=.0002)
        nn.init.trunc_normal_(self.target_temp_bias_line,  std=.0002)
        nn.init.trunc_normal_(self.temp_target_bias_table, std=.0002)
        nn.init.trunc_normal_(self.target_temp_bias_table, std=.0002)


        self.temp_bias_index = self.getBiasIndex(temp_len,temp_len)
        self.target_bias_index = self.getBiasIndex(target_len,target_len)
        
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
        #attn = (q @ k.transpose(-2, -1)) * self.scale
        #attn = attn.softmax(dim=-1)#[B,heads,num_patch,num_patch]
        relative_position_bias = 0
        if self.use_bias:
            temp_relative_position_bias = self.relative_position_bias_table_temp[self.temp_bias_index.view(-1)].view(
                self.temp_len * self.temp_len, self.temp_len * self.temp_len, -1)#[64,64,8]
            target_relative_position_bias = self.relative_position_bias_table[self.target_bias_index.view(-1)].view(
                self.target_len * self.target_len, self.target_len * self.target_len, -1)#[256,256,8]

            temp_relative_position_bias = temp_relative_position_bias.permute(2, 0, 1).contiguous()
            target_relative_position_bias = target_relative_position_bias.permute(2, 0, 1).contiguous()

            temp_target_bias = self.temp_target_bias_table.expand(-1,-1,self.temp_len**2) + self.temp_target_bias_line.expand(-1,self.target_len**2,-1)
            target_temp_bias = self.target_temp_bias_table.expand(-1,-1,self.target_len**2) + self.target_temp_bias_line.expand(-1,self.temp_len**2,-1)

            relative_position_bias =torch.cat((torch.cat((temp_relative_position_bias,temp_target_bias),dim = 1),
                                        torch.cat((target_temp_bias,target_relative_position_bias),dim = 1)),
                                        dim = 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_position_bias
        attn = attn.softmax(dim=-1) #[B,heads,num_patch,num_patch]
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class PartNorm(nn.Module):
    def __init__(self,embed_dim,len) -> None:
        super().__init__()
        self.len = len
        self.temp_norm = nn.LayerNorm(embed_dim)
        self.targ_norm = nn.LayerNorm(embed_dim)
    def forward(self,x):
        temp = self.temp_norm(x[:,:self.len,:])
        targ = self.targ_norm(x[:,self.len:,:])
        return torch.cat([temp,targ],dim = 1)
class PartMlp(nn.Module):
    def __init__(self,in_features,hidden_features,act_layer = nn.GELU,drop = 0.,len = 16*16) -> None:
        super().__init__()
        self.len = len
        self.temp_mlp = Mlp(in_features=in_features, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
        self.targ_mlp = Mlp(in_features=in_features, hidden_features=hidden_features, act_layer=act_layer, drop=drop)
    def forward(self,x):
        temp = self.temp_mlp(x[:,0:self.len,:])
        targ = self.targ_mlp(x[:,self.len:,:])
        return torch.cat([temp,targ],dim = 1)

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 relative_bias = True,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 temp_len = 16,
                 target_len = 22,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 splitannt = False,
                 split = False):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) if not split else PartNorm(dim,len=temp_len**2)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,temp_len=temp_len,target_len=target_len,relative_bias=relative_bias)
        if splitannt:
            self.attn = Convattn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,temp_len=temp_len,target_len=target_len,relative_bias=relative_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) if not split else PartNorm(dim,len=temp_len**2)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio) \
            if not split else PartMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Tailattn(nn.Module):
    def __init__(self,embed_dim,num_heads,drop_ratio,use_conv,splitattn,temp_len,target_len,check_point = True) -> None:
        super().__init__()
        self.temp_num = temp_len**2
        self.check_point = check_point
        self.depth = 4
        dpr = [x.item() for x in torch.linspace(0,drop_ratio, self.depth)]
        # self.selfattn =  nn.Sequential(
        #     *[swinCrossBasic(embed_dim,num_heads,splitattn=splitattn,use_conv=use_conv,roll_type = i % 4, #0,1,2,3
        #     drop_path_ratio=dpr[i],temp_len=temp_len,target_len=target_len) for i in range(self.depth)]
        # )
        self.cross1 = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.cross2 = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)

        self.taself = nn.Sequential(
            *[Block(embed_dim,num_heads,relative_bias=False,temp_len=temp_len,target_len=target_len) for i in range(2)]
        )
        self.taself2 = nn.Sequential(
            *[Block(embed_dim,num_heads,relative_bias=False,temp_len=temp_len,target_len=target_len) for i in range(2)]
        )
        self.cross_mlp1 = Mlp(embed_dim,4*embed_dim,embed_dim) 
        self.cross_mlp2 = Mlp(embed_dim,4*embed_dim,embed_dim) 
        self.cross_norm1 = nn.LayerNorm(embed_dim)
        self.cross_norm2 = nn.LayerNorm(embed_dim)
        self.dummy = DummyLayer()
    def forward(self,temp,target):
        B,num_token,embed_dim = temp.shape
        x = torch.cat([temp,target],dim = 1)

        embed = self.taself(target) if not self.check_point else cp.checkpoint_sequential(self.taself,2,target)
        embed = embed + self.cross1(query = embed,key = x,value = x)[0]\
         if not self.check_point else embed + cp.checkpoint(self.cross1,embed,x,x)[0]
        embed = embed + self.cross_mlp1(self.cross_norm1(embed))

       
        embed = self.taself2(embed) if not self.check_point else cp.checkpoint_sequential(self.taself2,2,embed)
        embed = embed + self.cross2(query = embed,key = x,value = x)[0]\
            if not self.check_point else embed + cp.checkpoint(self.cross2,embed,x,x)[0]#256
        out = embed + self.cross_mlp2(self.cross_norm2(embed))

        return out,None

    def _forward(self,temp,target):
        temp = self.dummy(temp)
        target = self.dummy(target)
        if self.check_point:
            out,w = cp.checkpoint(self.forward_nocp,temp,target)
        else:
            out,w = self.forward_nocp(temp,target)
        return out,w
    def _forwardold(self,temp,target):
        #weight1
        target_ = target
        temp_ = temp.mul(self.window)
        target = cp.checkpoint(self.weight,temp_,target,target)[0]
        temp = self.temp_weight(target_,temp_,temp_)[0]

        #temp self
        temp = temp + self.temp_self(query = temp,key = temp,value = temp)[0]
        temp = temp + self.temp_endmlp(self.temp_endnorm(temp))
        #target self
        target = target + cp.checkpoint(self.targ_self,target,target,target)[0]
        target = target + self.targ_selfmlp(self.targ_selfnorm(target)) 

        #weight2
        target_ = target
        temp_ = temp.mul(self.window)
        target = cp.checkpoint(self.weight2,temp_,target,target)[0]
        temp = self.temp_weight2(target_,temp_,temp_)[0]

        temp_ = temp
        #temp target
        temp = temp + cp.checkpoint(self.temptarg_cross, temp,target,target)[0]
        temp = temp + self.temp_mlp(self.temp_norm(temp))#256,288
       
        #target temp
        target = target + self.targtemp_cross(query = target,key = temp_,value = temp_)[0]
        target = target + self.targ_endmlp(self.targ_endnorm(target))#484,288

        #temp target
        out ,weight = self.cross(query = target,key = temp,value = temp)#256
        out = target + self.targ_mlp(self.targ_norm(out))
        return out,weight
    
class BasePart(nn.Module):
    def __init__(self,funcBuild = buildAll,use_checkpoint = True) -> None:
        super().__init__()

        _,self.swin_target = funcBuild(use_checkpoint)
        
        self.use_checkpoint = use_checkpoint
        self.dummy = DummyLayer()
    def forward(self,temp,target):
        temp = self.dummy(temp)
        target = self.dummy(target)

        temp = self.swin_target.forward_temp(temp) 
        target = self.swin_target(target) #if not self.use_checkpoint else cp.checkpoint(self.swin_target,target)
        return temp,target

    def temp_feature(self,temp):
        return self.swin_target.forward_temp(temp)

    def target_feature(self,target):
        return self.swin_target(target)

class Online(nn.Module):
    def __init__(self,dim = 384)-> None:
        super().__init__()
        self.dim = dim
        # self.combine = nn.Linear(2*dim,dim,bias=False)
        self.combine = nn.Sequential(
            nn.Conv2d(dim,dim,5,padding = 2,bias = False,padding_mode='replicate'),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
            nn.Conv2d(dim,dim,3,padding=1,bias = False,padding_mode='replicate'),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
        )
    def forward(self,temp,target,bbox):
        target = self.clip_features(target,bbox)
        #target :[B,C,12,12] temp:[B,N,C]
        B,N,C = temp.shape
        temp = temp.transpose(-1,-2).reshape(B,C,12,12)
        temp = (temp + self.combine(target)).flatten(2,3).transpose(-1,-2)
        return temp

    def clip_features(self,target,bbox):
        bbox = bbox * 24
        x,y,w,h = bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]
        l = (w+h)/2
        l = ((w+l)*(h+l)).sqrt()
        x_min = torch.ceil(x-l/2).int()
        x_min[x_min<0] = 0
        y_min = torch.ceil(y-l/2).int()
        y_min[y_min<0] = 0
        x_max = (x + l/2).int()
        x_max[x_max>23] = 23
        y_max = (y + l/2).int()
        y_max[y_max>23] = 23
        B,N,C = target.shape
        target = target.transpose(-1,-2).reshape(B,C,24,24)
        ta = target[0,:,y_min[0]:y_max[0],x_min[0]:x_max[0]].unsqueeze(0)
        ta = self.resize_target(ta)
        for i in range(1,B):
            t = target[i,:,y_min[i]:y_max[i],x_min[i]:x_max[i]].unsqueeze(0)
            t = self.resize_target(t)
            ta = torch.cat([ta,t],dim = 0)
        return ta
    def resize_target(self,target):
        B,C,L,L = target.shape
        #[B,C,12,12]
        target = F.interpolate(target,(12,12))#.flatten(2,3).transpose(-1,-2)
        return target

class Cross(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio = 4) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim,num_heads,batch_first=True)
        self.mlp = Mlp(dim,dim*mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self,features,target):
        features = self.norm(features)
        target = self.norm(target)
        target = target +self.cross(target,features,features)[0]
        target = target + self.mlp(self.norm2(target))
        return target
    
class My(nn.Module):
    def __init__(self,
                 depth = 8,
                 embed_dim = 384,
                 num_heads = 8,
                 mlp_ratio = 2,
                 drop_ratio = 0.2,
                 temp_len = 16,
                 target_len = 22,
                 relative_bias = True,
                 use_conv_bias = True,
                 few_ffn = False,
                 useV2 = False,# use
                 two_cross = False,
                 pre_cross = False,
                 drop_target = False,
                 funcBuild = getbase,
                 use_checkpoint = True) -> None:
        super().__init__()
        self.basepart = BasePart(funcBuild,use_checkpoint=use_checkpoint)

        drop_ratio = drop_ratio
        dpr = [x.item() for x in torch.linspace(0,drop_ratio, depth)]

        self.use_checkpoint = use_checkpoint
        self.depth = depth
        self.targetlen = target_len
        self.templen = temp_len
        self.embed_dim = embed_dim
        self.act = nn.Sigmoid()
        self.feature = None
        self._update_name(depth,embed_dim,relative_bias,useV2,use_conv_bias,two_cross,drop_target,few_ffn,pre_cross)
        # self.online = Online(embed_dim)
        # self.dummy = DummyLayer()
        self.tail_layers = nn.ModuleList()
        base_ratio = 0
        length = 2
        tl = temp_len
        tal = target_len
        dim = embed_dim
        
        self.pre_cross = Cross(embed_dim,num_heads,mlp_ratio) if pre_cross else None
        
        
        for i_layer in range(length):
            drop_ratio = dpr[i_layer * depth//length - 1] + 0.1
            layers = ForwardBlock(
               embed_dim= dim,
               num_heads= num_heads,
               mlp_ratio= mlp_ratio,
               base_ratio= base_ratio,
               drop_ratio= drop_ratio,
               window_size= 8,#target_len//3,
               depth= depth,
               temp_len= tl,
               target_len= tal,
               swin_tlen= target_len,
               qkv_bias= True,
               relative_bias= relative_bias,
               useV2= useV2,
               use_checkpoint = (i_layer%2==0 and use_checkpoint) ,change_dim=i_layer<1,two_cross=two_cross,few_ffn=few_ffn,
               use_conv_bias=use_conv_bias)
            if i_layer<1:
                tl = tl//2
                tal = tal//2
                dim = int(embed_dim*1.5)
                num_heads = int(num_heads*1.5)
            self.tail_layers.append(layers)
            base_ratio = drop_ratio
        #[B,num,4],[B,1,4]
        self.head = Head(dim)#ConvHead(dim,target_len)
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.head2 = Head(embed_dim)
        self.norm21 = nn.LayerNorm(embed_dim)
        self.norm22 = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

        self.return2 = True
    def forward(self,temp,target,online = None):
        temp,target = self.basepart(temp,target)            
        B,num_token,embed_dim = temp.shape
        length = int(math.sqrt(num_token))//2

        features = torch.cat([temp,target],dim=1)

        out = target
        weights = []
        for layer in self.tail_layers:
            features,out ,w,drop_index= layer(features,out)
            weights.append(w)
            features,out = layer.downSample(features,out)

        out = self.norm(out)
        classes,box,_ = self.head(out)
        output = {'pred_boxes':box,'pred_logits':classes,'gaussian':0}
        # output['pred_weights'] = w
        output['drop_index'] = drop_index.cpu().numpy()
        return [output],[]#outputs1 ,outputs2

    def _fowardHead(self,target,out,return2 = True):
        B,N,C = out.shape
        head = self.head
        norm1 = self.norm
        norm2 = self.norm2
        if C == self.embed_dim:
            head = self.head2
            norm1 = self.norm21
            norm2 = self.norm22


        out = norm1(out)
        classes,box,_ = head(out)
        output = {'pred_boxes':box,'pred_logits':classes,'gaussian':0}

        output2 = []
        if return2:
            out2 = norm2(target)
            classes2,box2,_ = head(out2)
            output2 = {'pred_boxes':box2,'pred_logits':classes2,'gaussian':0}
        return output,output2
    def set_not_all(self):
        for layer in self.tail_layers:
            layer.set_not_all()


    @torch.no_grad()
    def template(self,temp):
        self.feature = self.basepart.temp_feature(temp)
        self.online_feature = None
    

    @torch.no_grad()
    def track(self,target):
        target = self.basepart.target_feature(target)
        B,num_token,embed_dim = self.feature.shape
        feature = self.feature

        features = torch.cat([feature,target],dim=1)
        out = target
        for layer in self.tail_layers:
            features,out = layer(features,out)
        
        out = self.norm(out)
 
        classes,box = self.head(out)

        
        output = {'pred_boxes':box,'pred_logits':classes}
        return output

    def _update_name(self,depth,embed_dim,relative_bias,useV2,use_conv_bias,two_cross,drop_targ,few_ffn,pre_cross):
        self.str = 'depth_'+str(depth)+'embed_'+str(embed_dim)
        if relative_bias == False and use_conv_bias == False:
            self.str += '_nobias'
            print('no bias')
        if relative_bias == True:
            self.str += '_bias'
            print('relative_bias')
        if useV2:
            self.str += '_v2'
            print('use v2')
        if use_conv_bias:
            self.str += '_cb'
            print('spin conv bias')
        if two_cross:
            self.str += '_two_cross'
            print('two cross')
        if drop_targ:
            self.str+= '_dropta'
            print('drop target')
        if few_ffn:
            self.str += 'fewffn'
            print('try less ffn')
        if pre_cross:
            self.str += 'prec'
            print('pre_cross')
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.001)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    def track_init(self):
        for layer in self.tail_layers:
            layer.track_init()
        print('bias inited')
        
def GetMyBase(use_checkpoint = True):
    
    model =  My(depth=4,embed_dim=384,num_heads=8,mlp_ratio=4,drop_ratio=0.2,use_checkpoint = use_checkpoint,
    temp_len=12,target_len=24,funcBuild=getbase    )
    model.str = model.str + '_base.pth'
    return model

def GetMyBase512(use_checkpoint = True):
    model =  My(depth=4,embed_dim=384,num_heads=8,mlp_ratio=4,drop_ratio=0.2,use_checkpoint = use_checkpoint,
    temp_len=16,target_len=32,funcBuild=getbase512)
    model.str = model.str + '_base512.pth'
    return model

def GetMySmall(use_checkpoint = True):
    model = My(depth=4,embed_dim=384,num_heads=8,mlp_ratio=4,drop_ratio = 0.2,use_checkpoint = use_checkpoint,
    temp_len=8,target_len=16,funcBuild=getsmall)
    model.str = model.str + '_small.pth'
    return model

def GetMyLarge(use_checkpoint = True,
               relative_bias = True,
              use_conv_bias = True):
    model = My(depth=4,embed_dim=512,num_heads=8,mlp_ratio=4,drop_ratio=0.2,use_checkpoint = use_checkpoint,
    temp_len=8,target_len=16,funcBuild=getlarge,
    relative_bias = relative_bias,
    use_conv_bias=use_conv_bias)
    model.str = model.str+'_large.pth'
    return model

def GetMyLarge384(use_checkpoint = True):
    model = My(depth=4,embed_dim=512,num_heads=8,mlp_ratio=4,drop_ratio=0.2,use_checkpoint = use_checkpoint,
    temp_len=12,target_len=24,funcBuild=getlarge384)
    model.str = model.str+'_large384.pth'
    return model

def load_pretrained(model,path):
    if path is None:
        return
    ld = torch.load(path)
    #predict = ld['model']
    #predict = {k.replace('layers.',''):v for k,v in predict.items()}
    model.basepart.swin_target.load_state_dict(ld['model'],strict = False)
    model.str = 'pret_' + model.str
    ld = None
    print('pretrained loaded')

def load_nomlp(model,path):
    if path is None:
        return
    ld = torch.load(path)
    predict = {k:v for k,v in ld['state_dict'].items() if 'temp_endmlp' or 'targ_endmlp' not in k}
    model.load_state_dict(predict)
    print('tail no mlp')

if __name__ == "__main__":
    model = My()
    img = torch.rand([1,3,256,256])
    tmp = torch.rand([1,3,128,128])
    output = model(tmp,img)
        