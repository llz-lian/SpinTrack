from modulefinder import Module
import torch
import torch.nn as nn
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

        self.conv_q = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False,groups=dim), 
            nn.BatchNorm2d(dim)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False,groups=dim), 
            nn.BatchNorm2d(dim)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False,groups=dim), 
            nn.BatchNorm2d(dim)
        )

        self.proj = nn.Linear(dim,dim)
        #self.selfproj = nn.Linear(dim,dim)
        
        self.use_bias = relative_bias
        self.temp_len = temp_len
        self.target_len = target_len
        self.num_token = self.temp_len**2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.target_len - 1) * (2 * self.target_len - 1), num_heads))
        self.relative_position_bias_table_temp = nn.Parameter(
            torch.zeros((2 * self.temp_len - 1) * (2 * self.temp_len - 1), num_heads)
        )
        self.temp_target_bias_table = nn.Parameter(
            torch.zeros(num_heads,self.temp_len**2,1)
        )
        self.target_temp_bias_table = nn.Parameter(
            torch.zeros(num_heads,self.target_len**2,1)
        )
        self.temp_target_bias_line = nn.Parameter(
            torch.zeros(num_heads,1,self.temp_len**2)
        )
        self.target_temp_bias_line = nn.Parameter(
            torch.zeros(num_heads,1,self.target_len**2)
        )
        nn.init.trunc_normal_(self.temp_target_bias_line,  std=.0002)
        nn.init.trunc_normal_(self.target_temp_bias_line,  std=.0002)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.0002)
        nn.init.trunc_normal_(self.relative_position_bias_table_temp, std=.0002)
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

    def forward(self,x):
        B, N, C = x.shape
        #[B,nt+ns,dim] split
        target = x[:,self.num_token:,:]
        temp = x[:,0:self.num_token,:]
        #q:[B,ns,dim] kv:[B,nt+ns,dim*2]
        B,Ns,C = target.shape

        q,k,v = self.forward_conv(temp = temp,target = target)
        q = self.q(q).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        k = self.k(k).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        v = self.v(v).reshape(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        

        relative_position_bias = 0.
        if self.use_bias:
            temp_relative_position_bias = self.relative_position_bias_table_temp[self.temp_bias_index.view(-1)].view(
                self.temp_len * self.temp_len, self.temp_len * self.temp_len, -1)#[64,64,8]
            target_relative_position_bias = self.relative_position_bias_table[self.target_bias_index.view(-1)].view(
                self.target_len * self.target_len, self.target_len * self.target_len, -1)#[256,256,8]

            temp_relative_position_bias = temp_relative_position_bias.permute(2, 0, 1).contiguous()
            target_relative_position_bias = target_relative_position_bias.permute(2, 0, 1).contiguous()

            #[temp,target] [target,temp]
            target_temp_bias = self.temp_target_bias_table.expand(-1,-1,self.target_len**2) + self.target_temp_bias_line.expand(-1,self.temp_len**2,-1)
            temp_target_bias = self.target_temp_bias_table.expand(-1,-1,self.temp_len**2) + self.temp_target_bias_line.expand(-1,self.target_len**2,-1)

            relative_position_bias =torch.cat((torch.cat((temp_relative_position_bias,target_temp_bias),dim = 2),
                                        torch.cat((temp_target_bias,target_relative_position_bias),dim = 2)),
                                        dim = 1).unsqueeze(0)
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
    def forward_conv(self,temp,target):
        #[B,n,dim] =>[B,dim,n] => [B,dim,len,len]
        B,_,dim = temp.shape
        temp = temp.transpose(-1,-2).reshape(B,dim,self.temp_len,self.temp_len)
        target = target.transpose(-1,-2).reshape(B,dim,self.target_len,self.target_len)
        
        qtm = self.conv_q(temp).flatten(2,3).transpose(-1,-2)
        ktm = self.conv_k(temp).flatten(2,3).transpose(-1,-2)
        vtm = self.conv_v(temp).flatten(2,3).transpose(-1,-2)

        qta = self.conv_q(target).flatten(2,3).transpose(-1,-2)
        kta = self.conv_k(target).flatten(2,3).transpose(-1,-2)
        vta = self.conv_v(target).flatten(2,3).transpose(-1,-2)

        q = torch.cat([qtm,qta],dim = 1)
        k = torch.cat([ktm,kta],dim = 1)
        v = torch.cat([vtm,vta],dim = 1)

        return q,k,v

class SplitAttn(nn.Module):
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
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)       

        self.conv_q = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False,groups=dim), 
            nn.BatchNorm2d(dim)
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False,groups=dim), 
            nn.BatchNorm2d(dim)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=3,padding=1,bias=False,groups=dim), 
            nn.BatchNorm2d(dim)
        )

        self.proj = nn.Linear(dim,dim)
        #self.selfproj = nn.Linear(dim,dim)
        
        self.use_bias = relative_bias
        self.temp_len = temp_len
        self.target_len = target_len
        self.num_token = self.temp_len**2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.target_len - 1) * (2 * self.target_len - 1), num_heads))
        self.relative_position_bias_table_temp = nn.Parameter(
            torch.zeros((2 * self.temp_len - 1) * (2 * self.temp_len - 1), 1)
        )
        self.temp_target_bias_table = nn.Parameter(
            torch.zeros(num_heads,1,self.temp_len**2)
        )
        self.target_temp_bias_table = nn.Parameter(
            torch.zeros(num_heads,self.target_len**2,1)
        )

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.0002)
        nn.init.trunc_normal_(self.relative_position_bias_table_temp, std=.0002)
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

    def forward(self,x):
        B, N, C = x.shape
        #[B,nt+ns,dim] split
        target = x[:,self.num_token:,:]
        temp = x[:,0:self.num_token,:]
        #q:[B,ns,dim] kv:[B,nt+ns,dim*2]
        B,Ns,C = target.shape

        q,k,v = self.forward_conv(temp = temp,target = target)
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        #temp
        qtm = q[:,0:self.num_token,:].reshape(B, N-Ns, self.num_heads, C//self.num_heads).transpose(1,2)
        ktm = k[:,0:self.num_token,:].reshape(B, N-Ns, self.num_heads, C//self.num_heads).transpose(1,2)
        vtm = v[:,0:self.num_token,:].reshape(B, N-Ns, self.num_heads, C//self.num_heads).transpose(1,2)

        #target
        qta = q[:,self.num_token:,:].reshape(B, Ns, self.num_heads, C//self.num_heads).transpose(1,2)
        k = k.reshape(B,N,self.num_heads, C//self.num_heads).transpose(1,2)
        v = v.reshape(B,N,self.num_heads, C//self.num_heads).transpose(1,2)

        relative_position_bias = 0.
        temp_relative_position_bias = 0.
        if self.use_bias:
            target_relative_position_bias = self.relative_position_bias_table[self.target_bias_index.view(-1)].view(
                self.target_len * self.target_len, self.target_len * self.target_len, -1)#[heads,nt,nt]
            temp_relative_position_bias = self.relative_position_bias_table_temp[self.temp_bias_index.view(-1)].view(
                self.temp_len * self.temp_len, self.temp_len * self.temp_len, -1)
            
            temp_relative_position_bias = temp_relative_position_bias.permute(2, 0, 1).contiguous()
            
            target_relative_position_bias = target_relative_position_bias.permute(2, 0, 1).contiguous()

            padding = self.temp_target_bias_table.expand(-1,self.target_len**2,-1) + self.target_temp_bias_table.expand(-1,-1,self.temp_len**2)
            relative_position_bias = torch.cat([padding,target_relative_position_bias],dim = 2)
        #[B,heads,Ns,dim_heads] * [B,heads,dim_per_head,Nt+ns] ==> [B,heads,Ns,Nt+ns]
        qkt = qta @ k.transpose(-1,-2) * self.scale + relative_position_bias
        qkt = qkt.softmax(dim = -1)
        qkt = self.attn_drop(qkt)
        #[B,heads,Ns,Nt+ns] * [B,heads,Nt+ns,dim_per_head] ==> [B,heads,Ns,dim_per_head]
        v = (qkt @ v).transpose(1, 2).reshape(B, Ns, C)

        #temp self
        qkt = qtm @ ktm.transpose(-1,-2) * self.scale + temp_relative_position_bias
        qkt = qkt.softmax(dim = -1)
        qkt = self.attn_drop(qkt)
        temp = (qkt @ vtm).transpose(1, 2).reshape(B, N-Ns, C)

        x = torch.cat([temp,v],dim = 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def forward_conv(self,temp,target):
        #[B,n,dim] =>[B,dim,n] => [B,dim,len,len]
        B,_,dim = temp.shape
        temp = temp.transpose(-1,-2).reshape(B,dim,self.temp_len,self.temp_len)
        target = target.transpose(-1,-2).reshape(B,dim,self.target_len,self.target_len)
        
        qtm = self.conv_q(temp).flatten(2,3).transpose(-1,-2)
        ktm = self.conv_k(temp).flatten(2,3).transpose(-1,-2)
        vtm = self.conv_v(temp).flatten(2,3).transpose(-1,-2)

        qta = self.conv_q(target).flatten(2,3).transpose(-1,-2)
        kta = self.conv_k(target).flatten(2,3).transpose(-1,-2)
        vta = self.conv_v(target).flatten(2,3).transpose(-1,-2)

        q = torch.cat([qtm,qta],dim = 1)
        k = torch.cat([ktm,kta],dim = 1)
        v = torch.cat([vtm,vta],dim = 1)

        return q,k,v