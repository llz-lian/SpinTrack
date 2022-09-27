from .splitattn_win import Mlp,swinCrossBasic
from .convhead import DoubleHead
from .swin import getspintrack
import torch
import torch.nn as nn
class spinTrack(nn.Module):
    def __init__(self,use_checkpoint = True,num_heads = 8,embed_dim = 384,temp_len = 12,target_len = 24,backbone = getspintrack,depth = 8) -> None:
        super().__init__()
        self.temp_len = temp_len
        self.checkpoint = False#use_checkpoint
        self.backbone = backbone(False)
        self.blocks = nn.ModuleList()
        for i_layer in range(depth):
            layer = swinCrossBasic(embed_dim,num_heads,roll_type= i_layer%4,temp_len=temp_len,target_len=target_len,
            mlp_ratio=4,drop_path_ratio=0.1,qkv_bias=True,roll_kv = True)
            self.blocks.append(layer)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.decoder = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.mlp = Mlp(embed_dim,embed_dim*4)
        self.head = DoubleHead(embed_dim)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.str = 'spinT'
    def forward(self,temp,target,online = None):
        temp = self.backbone.forward_temp(temp)
        target = self.backbone(target)
        features = torch.cat([temp,target],dim = 1)
        for layer in self.blocks:
            features = layer(features)
        features = self.norm1(features)
        target = features[:,self.temp_len**2:,:]
        target = target + self.decoder(self.norm1(target),features,features)[0]
        target = target + self.mlp(self.norm2(target))
        target = self.head_norm(target)
        cls,reg,_ =self.head(target)
        output = {'pred_boxes':reg,'pred_logits':cls}
        return [output],[]
    def set_not_all(self):
        pass
def build_spin(use_checkpoint = True,**kwargs):
    return spinTrack(use_checkpoint = use_checkpoint)