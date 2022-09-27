from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
import math
import torch.nn.functional as F
import copy


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
        self .groups = groups
        self.ratio = reduce_ratio
        self.dilation = dilation #?
        self.padding = padding
        self.bias = bias

        self.tnum = math.ceil( float(channels)/reduce_ratio)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels,self.tnum,kernel_size=1,stride=stride),
            nn.BatchNorm2d(self.tnum),
            nn.PReLU()
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
        #self.attn = nn.MultiheadAttention(in_features,num_heads=num_heads,add_bias_kv=qkv_bias)
        self.attn = Attention(dim=in_features,num_heads=num_heads,qkv_bias=qkv_bias)
        hidden = in_features * mlp_ratio
        self.mlp = Mlp(in_features, hidden_features= hidden,drop= drop)
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)

    def forward(self,features):
        features = self.norm1(features + self.attn(features))
        features = self.norm2(features + self.mlp(features))
        return features




class DecoderBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,batch_first = True)
        hidden = embed_dim * mlp_ratio
        self.Mlp = Mlp(embed_dim,hidden_features=hidden)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self,key,query):
        output = self.norm1(query + self.attn(query = query,key = key,value = key)[0])
        output = self.norm2(output + self.Mlp(output))
        return output
class EncoderBlock(nn.Module):
    def __init__(self,in_channel = 3,mid_channel = 32,out_channel = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,mid_channel,kernel_size=1),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 =nn.Sequential(
            nn.Conv2d(mid_channel,out_channel,kernel_size = 1),
            nn.BatchNorm2d(out_channel)
        )

        self.invo = nn.Sequential(
            RealInvo(mid_channel),
            RealInvo(mid_channel),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.invo(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads,mlp_ratio,depth):
        super().__init__()
        decoder = DecoderBlock(embed_dim, num_heads,mlp_ratio)
        self.layers = _get_clones(decoder, depth)
    def forward(self,q,memory):
        output = q
        for layer in self.layers:
            output = layer(key = memory, query = output)
        return output


class Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = layer(x)
        return x


class Imgtotoken(nn.Module):
    def __init__(self,img_size,embed_dim,patch_size,
                 in_channel = 3,mid_ratio = 1,out_channels = [3,3,3]):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.blocks = self.buildBlock()
        self.temp_embed = Embed(img_size=img_size,embed_dim=embed_dim,in_channel=out_channels[-1],patch_size=patch_size)
    def forward(self,x):
        out = self.blocks(x)
        out = self.temp_embed(out)
        return out
    def buildBlock(self):
        blocks = []
        in_channel = self.in_channel
        for out_channel in self.out_channels:
            mid_channel = self.mid_ratio * out_channel
            blocks.append(EncoderBlock(in_channel,mid_channel,out_channel))
            in_channel = out_channel
        return nn.Sequential(*blocks)


class My(nn.Module):
    def __init__(self,
    target_size = 256,
    template_size = 112,
    in_channel = 3,
    embed_dim = 512,
    num_heads = 8,
    depth = 8,
    num_addon = 4,
    target_patch = 32,
    template_patch = 16,
    mlp_ratio = 3,
    drop = 0.,
    num_classes = 1,
    en_depth = 4):
        super().__init__()
        self.temp_embed = Imgtotoken(template_size,embed_dim,template_patch,out_channels=[3,6,9])
        self.target_embed = Imgtotoken(target_size,embed_dim,target_patch,out_channels = [3,3]) 
        
        self.token_blocks = nn.Sequential(*[Block(embed_dim,num_heads,mlp_ratio=1,drop = drop) for i in range(en_depth)])
        self.tar_blocks = nn.Sequential(*[Block(embed_dim,num_heads,mlp_ratio=1,drop = drop) for i in range(en_depth)])
                
        self.blocks = nn.Sequential(*[Block(embed_dim,num_heads,mlp_ratio=mlp_ratio,drop = drop) for i in range(depth)])

        self.token_grid_len = template_size//template_patch
        self.num_token = self.token_grid_len **2
        self.num_in = (self.token_grid_len * (self.token_grid_len+1))//2
        self.num_addon = num_addon
        self.num_patch = (target_size//target_patch)**2

        self.token_pos = nn.Parameter(torch.zeros(1,self.num_token,embed_dim))
        self.target_pos = nn.Parameter(torch.zeros(1,self.num_patch,embed_dim))
        self.pos = nn.Parameter(torch.zeros(1,self.num_patch + self.num_token + self.num_addon,embed_dim))
        self.addon = nn.Parameter(torch.zeros(1,num_addon,embed_dim))

        self.target_norm = nn.LayerNorm(embed_dim)
        self.temp_norm = nn.LayerNorm(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.reg = Regression(embed_dim,embed_dim,num_classes+1,3)
        self.regpos = Regression(embed_dim,embed_dim,4,3)


    def forward(self,target,tp):
        addon_tokens = self.addon.expand(target.shape[0],-1,-1)

        target = self.target_embed(target)#[B,HW,C] B,64,512
        target = target + self.target_pos
        target = self.target_norm(target)
        target = self.tar_blocks(target)

        tp = self.temp_embed(tp)#[B,HW,C] B,16,512
        tp = tp + self.token_pos
        tp = self.temp_norm(tp)
        tp = self.token_blocks(tp)

        #tokens = torch.cat((tp,addon_tokens),dim=1)
        #output = self.decoder(q = tokens,memory = target)
        features = torch.cat((target,tp,addon_tokens),dim = 1)
        features = features + self.pos
        features = self.norm(features)
        features = self.blocks(features)
        features = features[:,-(self.num_addon+self.num_token):,:]
        
        box = self.regpos(features).sigmoid()
        classes = self.reg(features)
        output = {'pred_boxes':box,'pred_logits':classes}

        return output


if __name__ == "__main__":
    modle = My()
    target = torch.randn(2,3,256,256)
    template = torch.randn(2,3,112,112)
    classes,box = modle(target,template)
    classes = classes.squeeze(0)
    box = classes.squeeze(0)
    print(box)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])