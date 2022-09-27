import torch
import torch.nn as nn
import numpy as np
# from torchinfo import summary
class Weight(nn.Module):
    def __init__(self,embed_dim,num_heads,len_q,len_k,len_v,return_qkt) -> None:
        super().__init__()
        assert len_k == len_v, 'k,v dim'
        self.len_q = len_q
        self.len_k = len_k
        self.len_v = len_v
        self.returnqkt = return_qkt
        self.wh = int(self.len_q ** 0.5)
        self.group = int(self.len_k ** 0.5)
        assert self.wh **2 == self.len_q,'not sq'
        self.weight1 = nn.Sequential(
            nn.Linear(self.len_k,self.len_k,bias=False),
            nn.GELU(),
            nn.Linear(self.len_k,self.len_k,bias = False),
            nn.GELU(),
            nn.Linear(self.len_k,1,bias=False)
        )
        #self.wconv3 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1,bias=True,padding_mode='replicate')
        self.act = nn.Tanh()
        self.norm = nn.LayerNorm(len_k,eps=1e-7,elementwise_affine=False)

    def forward(self,query,key,value):
        B,_,C = query.shape
        with torch.no_grad():#inf and nan
        #[B,len_q,embed_dim] * [B,len_v,embed_dim] ==> [B,len_q,len_k]
            q = query/100
            k = key/100
            ll_query_ll = q.square().sum(dim = -1).sqrt().unsqueeze(-1)#[B,len_q,1]
            ll_key_ll = k.square().sum(dim = -1).sqrt().unsqueeze( 1)#[B,1,len_k]
            outer = ll_query_ll @ ll_key_ll#[B,len_q,len_k]
            upper = q @ k.transpose(-1,-2)
            #[B,len_q,len_k]
            similar_rev = upper/(outer + 1e-6)
            #similar_rev = self.norm(similar_rev)
        if torch.any(torch.isnan(outer)) or torch.isinf(outer).any():
            print('nan')
        if torch.any(torch.isnan(upper)) or torch.isinf(upper).any():
            print('nan')
        if torch.any(torch.isnan(similar_rev)):
            print('nan')
        #[B,len_q,len_k] => [B,len_q,1] => [B,1,len_q]
        weight = self.weight1(similar_rev).transpose(-1,-2)

        #[B,1,len_q] @ [B,len_q,len_k] => [B,1,len_k] 
        weight = (weight @ similar_rev) / self.wh

        weight = weight.transpose(-1,-2)
        weight = self.act(weight)
        #[B,len_k,1] mul [B,len_v,embed_dim]
        out = value.mul(weight)
        #[B,len_v,embed_dim] => [B*embed_dim,1,wh,wh]
        # out = out.transpose(-1,-2)
        # out = out.flatten(0,1).reshape(B*C,1,self.group,self.group)
        # out = self.proj(out)
        # out = out.reshape(B,C,self.group,self.group).flatten(2,3).transpose(-1,-2)
        out = out + value
        if self.returnqkt:
            return out,similar_rev
        return out
# import torch
# import torch.nn as nn
# # from torchinfo import summary
# class Weight(nn.Module):
#     def __init__(self,embed_dim,num_heads,len_q,len_k,len_v,return_qkt) -> None:
#         super().__init__()
#         assert len_k == len_v, 'k,v dim'
#         self.len_q = len_q
#         self.len_k = len_k
#         self.len_v = len_v
#         self.returnqkt = return_qkt
#         #self.qkt_bias = nn.Parameter(torch.zeros([1,self.num_heads,len_q,len_k]))
#         self.wh = int(self.len_q ** 0.5)
#         self.group = int(self.len_k ** 0.5)
#         assert self.wh **2 == self.len_q,'not sq'
#         self.weight1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.len_k,out_channels=self.len_k * 2,kernel_size=1,padding=0,groups= self.len_k),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=self.len_k * 2,out_channels=1,kernel_size=3,padding=1)
#             #nn.Tanh(
#         )
#         self.proj = nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim,kernel_size=3,padding=1,groups=num_heads)
#         self.act = nn.Tanh()
#     def forward(self,query,key,value):
#         B,_,C = query.shape
#         with torch.no_grad():
#         #[B,len_q,embed_dim] * [B,len_v,embed_dim] ==> [B,len_q,len_k]
#             ll_query_ll = query.square().sum(dim = -1).sqrt().unsqueeze(-1)#[B,len_q,1]
#             ll_key_ll = key.square().sum(dim = -1).sqrt().unsqueeze( 1)#[B,1,len_k]
#             outer = ll_query_ll @ ll_key_ll#[B,len_q,len_k]
#             upper = query @ key.transpose(-1,-2)
#             #[B,len_q,len_k]
#             similar_rev = upper/(outer + 1e-6)

#         #[B,len_q,len_k] => [B,len_k,wh,wh] =>[B,1,wh,wh] => [B,1,len_q]
#         weight = self.weight1(similar_rev.transpose(-1,-2).reshape(B,self.len_k,self.wh,self.wh))
#         weight = weight.flatten(2,3)
#         #[B,1,len_q] @ [B,len_q,len_k] => [B,1,len_k] => [B,len_k,1]
#         weight = (weight @ similar_rev) / self.len_q
#         weight = weight.transpose(-1,-2)
#         weight = self.act(weight)
#         #[B,len_k,1] mul [B,len_v,embed_dim]
#         out = self.proj(value.mul(weight).transpose(-1,-2).reshape(B,C,self.group,self.group)).flatten(2,3).transpose(-1,-2)
#         out = out + value
#         if self.returnqkt:
#             return out,similar_rev
#         return out



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# #from torchinfo import summary
# class Weight(nn.Module):
#     def __init__(self,embed_dim,num_heads,len_q,len_k,len_v,return_qkt) -> None:
#         super().__init__()
#         self.q = nn.Linear(embed_dim,embed_dim)
#         self.k = nn.Linear(embed_dim,embed_dim)
#         self.v = nn.Linear(embed_dim,embed_dim)
#         #self.qkv = nn.Linear(embed_dim,embed_dim)
#         assert len_k == len_v, 'k,v dim'
#         self.len_q = len_q
#         self.len_k = len_k
#         self.len_v = len_v
#         self.num_heads = num_heads
#         self.sacle = self.num_heads ** -0.5
#         self.dim_per_head = embed_dim//num_heads
#         self.returnqkt = return_qkt
#         assert num_heads * self.dim_per_head == embed_dim ,'heads nope'
#         self.proj = nn.Linear(embed_dim,embed_dim)
#         #self.qkt_bias = nn.Parameter(torch.zeros([1,self.num_heads,len_q,len_k]))
#         self.wh = int(self.len_q ** 0.5)
#         self.weight = nn.Sequential(
#             nn.Conv2d(in_channels=embed_dim,out_channels=embed_dim*2,kernel_size=1,groups=self.dim_per_head),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=embed_dim*2,out_channels=embed_dim,kernel_size=3,padding=1,groups=self.dim_per_head),
#             nn.Sigmoid()
#         )
#         self.norm = nn.LayerNorm(self.dim_per_head)
#         self.qktproj = nn.Sequential(
#             nn.Linear(self.dim_per_head,self.dim_per_head*4),
#             nn.GELU(),
#             nn.Linear(4*self.dim_per_head,self.dim_per_head)
#         )
#         assert self.wh **2 == self.len_q,'not sq'
#     def forward(self,query,key,value):
#         B,_,C = query.shape

#         #[B,len,embed_dim]
#         q = self.q(query)
#         #[B,len_q,embed_dim]=>[B,embed_dim,len_q] =>B[B,embed_dim,wh,wh] =>[B,embed_dim,wh,wh]
#         weight = self.weight(q.transpose(-1,-2).reshape(B,C,self.wh,self.wh))
#         #[B,embed_dim,wh,wh]=>[B,num_heads,embed_per_head,len_q]
#         weight = weight.flatten(2,3).reshape(B,self.num_heads,self.dim_per_head,self.len_q)

#         #[B,num_head,len,embed_per_head]
#         q = q.reshape(B,self.num_heads,self.len_q,self.dim_per_head)
#         k = self.k(key).reshape(B,self.num_heads,self.len_k,self.dim_per_head)
#         v = self.v(value).reshape(B,self.num_heads,self.len_v,self.dim_per_head)

#         #[B,num_heads,len_q,len_k]
#         qkt = (q @ k.transpose(-1,-2)) * self.sacle #+ self.qkt_bias
#         qkt = qkt.softmax(dim = -1)
#         #[B,num_heads,embed_per_head,len_q] * [B,num_heads,len_q,len_k] =>[B,num_heads,embed_per_head,len_k]
#         qkt = (weight @ qkt) * self.sacle 
#         #[B,num_heads,embed_per_head,len_k] =>[B,num_heads,1,len_k]=>[B,num_heads,len_k,1]
#         #qkt = qkt.sum(dim = 2)/self.dim_per_head
#         qkt = qkt.transpose(-1, -2)
#         qkt = qkt + self.qktproj(self.norm(qkt))
#         #qkt = qkt.unsqueeze(dim = 2)
#         #[B,num_heads,len_k,1] =>[B,num_heads,len_k,embed_per_head]
#         #qkt = qkt.repeat(1,1,1,self.dim_per_head)

#         #len_k = len_v
#         #[B,num_heads,len_v,embed_per_head] dot [B,num_heads,len_v,embed_per_head]
#         #[B,len_v,num_heads,embed_per_head]
#         #[B,len_v,embed_dim]
#         out = v.mul(qkt).transpose(1,2).reshape(B, self.len_v, C)#28,28,8,484,36
#         out = self.proj(out) + value
#         if self.returnqkt:
#             return out,qkt
#         return out

if __name__ == '__main__':
    input_q = torch.ones([8,256,288])
    input_k = torch.ones([8,484,288])
    input_v = torch.ones([8,484,288])
    weight = Weight(288,8,256,484,484,False)
    # summary(weight,input_data=(input_q,input_k,input_v))
    out = weight(input_q,input_k,input_v)
    # #print(out.shape)
    # total = sum([param.nelement() for param in weight.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))