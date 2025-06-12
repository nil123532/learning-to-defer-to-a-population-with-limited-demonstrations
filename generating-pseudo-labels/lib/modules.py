import torch.nn as nn
import functools
import torch
import sys
import torch.nn.functional as F
import math
import copy


class MultiHeadAttn(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.fc_q = nn.Linear(dim_q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_out, bias=False)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)
        
    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, -1), -3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, -3), -1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = [self.scatter(x) for x in [q, k, v]]
        A_logits = q_ @ k_.transpose(-2, -1) / math.sqrt(self.dim_out)
        if mask is not None:
            mask = mask.bool().to(q.device)
            mask = torch.stack([mask]*q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, -3)
            A = torch.softmax(A_logits.masked_fill(mask, -float('inf')), -1)
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask=mask))
        out = self.ln2(out + F.relu(self.fc_out(out)))
        return out

class SelfAttn(MultiHeadAttn):
    def __init__(self, dim_in, dim_out, num_heads=8):
        super().__init__(dim_in, dim_in, dim_in, dim_out, num_heads)

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)


class LinearNN(nn.Module):
    def __init__(self, num_classes, low_dim=64, feature_dim=1280, proj=False):
        super().__init__()
        self.proj= proj
        self.linear_layers = nn.Sequential(nn.Linear(feature_dim, num_classes, bias=True))
        if proj:
            self.l2norm = Normalize(2)

            self.fc1 = nn.Linear(feature_dim, feature_dim)
            self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
            self.fc2 = nn.Linear(feature_dim, low_dim)

    def forward(self, feat):
        out = self.linear_layers(feat)
        if self.proj:
            feat = self.fc1(feat)
            feat = self.relu_mlp(feat)
            feat = self.fc2(feat)

            feat = self.l2norm(feat)
            return out, feat
        else:
            return out
        
def build_mlp(dim_in, dim_hid = 128, dim_out = 1, depth = 4, activation='relu'):
    act = get_activation(activation)
    if depth==1:
        modules = [nn.Linear(dim_in, dim_out)] # no hidden layers
    else: # depth>1
        modules = [nn.Linear(dim_in, dim_hid), act()]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(act())
        modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


def get_activation(act_str):
    if act_str == 'relu':
        return functools.partial(nn.ReLU, inplace=True)
    elif act_str == 'elu':
        return functools.partial(nn.ELU, inplace=True)
    else:
        raise ValueError('invalid activation')


class ClassifierRejectorWithContextEmbedder(nn.Module):
    def __init__(self, num_classes = 2, n_features=None, dim_hid=128, depth_embed=6, dim_class_embed=128,with_softmax=True,actual_classes=10,with_attn=False):
        super(ClassifierRejectorWithContextEmbedder, self).__init__()
        self.num_classes = num_classes
        self.with_attn = with_attn
        self.with_softmax = with_softmax
        self.dim_hid = dim_hid
        self.n_features = n_features #same as the output features of the embedding model
        
        self.fc = nn.Linear(n_features+dim_hid, num_classes)
        self.fc.bias.data.zero_()

        self.embed_class_m = nn.Embedding(actual_classes, dim_class_embed) # created a class embedding layer
        self.embed_class = nn.Embedding(actual_classes, dim_class_embed) # created a class embedding layer


        if self.with_attn == 'mlp':
            self.embed = build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed)
        elif self.with_attn == 'attn':
            self.embed = nn.Sequential(
                build_mlp(n_features+dim_class_embed*2, dim_hid, dim_hid, depth_embed-2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid)
            )
        else:
            self.embed = None #if single is chosen 
        
        if with_attn == 'attn':
            self.attn = MultiHeadAttn(n_features, n_features, dim_hid, dim_hid)

    def forward(self, x_embed, cntxt=None):
        '''
        Args:
            x : tensor [B,3,32,32]
            cntxt : AttrDict, with entries
                xc : tensor [E,Nc,3,32,32]
                yc : tensor [E,Nc]
                mc : tensor [E,Nc]
        '''
        if cntxt is None or self.with_attn == 'single':
            n_experts = 1
        else:
            n_experts = cntxt.xc.shape[0]
        
        if cntxt is None or self.with_attn == 'single':
            embedding = torch.zeros((n_experts, x_embed.shape[0], self.dim_hid), device=x_embed.device)

        else:
            embedding = self.encode(cntxt, x_embed) # [E,B,H]

        
        x_embed = x_embed.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]

        packed = torch.cat([x_embed,embedding], -1) # [E,B,Dx+H]
        
        out = self.fc(packed) # [E,B,2] 
        
        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out
    
    def encode(self, cntxt, xt):
        n_experts = cntxt.xc.shape[0]
        batch_size = xt.shape[0]

     
        xc_embed = cntxt.em # [E,Nc,Dx]
   

        yc_embed = self.embed_class(cntxt.yc) # [E,Nc,H]
        mc_embed = self.embed_class_m(cntxt.mc) # [E,Nc,H]
        out = torch.cat([xc_embed,yc_embed,mc_embed], -1) # [E,Nc,Dx+2H]


        out = self.embed(out) # [E,Nc,H]

        if not self.with_attn:
            embedding = out.mean(-2) # [E,H]
            embedding = embedding.unsqueeze(1).repeat(1,batch_size,1) # [E,B,H]
        else:
            xt = xt.unsqueeze(0).repeat(n_experts,1,1) # [E,B,Dx]
            embedding = self.attn(xt, xc_embed, out)
        
        return embedding


class ClassifierRejector(nn.Module):
    def __init__(self, num_classes, n_features, with_softmax=True, decouple=False):
        super(ClassifierRejector, self).__init__()
  

        self.fc_clf = nn.Linear(n_features, num_classes)
        self.fc_clf.bias.data.zero_()

        self.fc_rej = nn.Linear(n_features, 1)
        self.fc_rej.bias.data.zero_()

        self.with_softmax = with_softmax
    

    def forward(self, x):
        out = self.base_model_clf(x)
        logits_clf = self.fc_clf(out) # [B,K]

        out = self.base_model_rej(x)
        logit_rej = self.fc_rej(out) # [B,1]

        out = torch.cat([logits_clf,logit_rej], -1) # [B,K+1]

        if self.with_softmax:
            out = F.softmax(out, dim=-1)
        return out









      
class DeepNN(nn.Module):
    def __init__(self,num_classes, low_dim=64, feature_dim=1280, proj=False):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),        # optional, for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, feat):
        return self.layers(feat)