from pprint import pprint
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math as m
import torch
from easydict import EasyDict
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

from cqt_nsgt_pytorch import CQT_nsgt
# Patch the nsgfwin function used inside cqt_nsgt_pytorch to avoid a numpy
# casting error when clipping integer arrays.

import torchaudio
import einops
import math

"""
As similar as possible to the original CQTdiff architecture, but using the octave-base representation of the CQT
This should be more memory efficient, and also more efficient in terms of computation, specially when using higher sampling rates.
I am expecting similar performance to the original CQTdiff architecture, but faster. 
Perhaps the fact that I am using powers of 2 for the time sizes is critical for transient reconstruction. I should thest CQT matrix model with powers of 2, this requires modifying the CQT_nsgt_pytorch.py file.
"""
def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

class Conv1d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel=1, bias=False, dilation=1,
        init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel, fan_out=out_channels*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel], **init_kwargs) * init_weight) 
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        #f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        #print(x.shape, w.shape)
        if w is not None:
                x = torch.nn.functional.conv1d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))
        return x
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel=(1,1), bias=False, dilation=1,
        init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel[0]*kernel[1], fan_out=out_channels*kernel[0]*kernel[1])
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel[0], kernel[1]], **init_kwargs) * init_weight) 
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        #f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding="same", dilation=self.dilation)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 1e-4, channel_last=True):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

class BiasFreeLayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-7):
        super(BiasFreeLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,1,num_features))
        #self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        #self.beta = torch.zeros(1,num_features,1,1)
        self.eps = eps

    def forward(self, x):
        N, T, C = x.size()
        #x = x.view(N, self.num_groups ,-1,H,W)
        #x=einops.rearrange(x, 'n t c -> n (t c)')
        #mean = x.mean(-1, keepdim=True)
        #var = x.var(-1, keepdim=True)

        std=x.std(-1, keepdim=True) #reduce over channels and time
        #var = x.var(-1, keepdim=True)

        ## normalize
        x = (x) / (std+self.eps)
        # normalize
        #x=einops.rearrange(x, 'n (t c) -> n t c', t=T)
        #x = x.view(N,C,H,W)
        return x * self.gamma

class BiasFreeGroupNorm(nn.Module):

    def __init__(self, num_features, num_groups=32, eps=1e-7):
        super(BiasFreeGroupNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.beta = nn.Parameter(torch.zeros(1,num_features,1,1))
        #self.beta = torch.zeros(1,num_features,1,1)
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, F, T = x.size()
        #x = x.view(N, self.num_groups ,-1,H,W)
        gc=C//self.num_groups
        x=einops.rearrange(x, 'n (g gc) f t -> n g (gc f t)', g=self.num_groups, gc=gc)
        #mean = x.mean(-1, keepdim=True)
        #var = x.var(-1, keepdim=True)

        std=x.std(-1, keepdim=True) #reduce over channels and time
        #var = x.var(-1, keepdim=True)

        ## normalize
        x = (x) / (std+self.eps)
        # normalize
        x=einops.rearrange(x, 'n g (gc f t) -> n (g gc) f t', g=self.num_groups, gc=gc, f=F, t=T)
        #x = x.view(N,C,H,W)
        return x * self.gamma



class RFF_MLP_Block(nn.Module):
    """
        Encoder of the noise level embedding
        Consists of:
            -Random Fourier Feature embedding
            -MLP
    """
    def __init__(self, emb_dim=512, rff_dim=32, init=None):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, rff_dim]), requires_grad=False)
        self.MLP = nn.ModuleList([
            Linear(2*rff_dim, 128, **init),
            Linear(128, 256, **init),
            Linear(256, emb_dim, **init),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class AddFreqEncodingRFF(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim, N):
        super(AddFreqEncodingRFF, self).__init__()
        self.N=N
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, N]), requires_grad=False)


        self.f_dim=f_dim #f_dim is fixed
        embeddings=self.build_RFF_embedding()
        self.embeddings=nn.Parameter(embeddings, requires_grad=False) 

        
    def build_RFF_embedding(self):
        """
        Returns:
          table:
              (shape: [C,F], dtype: float32)
        """
        freqs = self.RFF_freq
        #freqs = freqs.to(device=torch.device("cuda"))
        freqs=freqs.unsqueeze(-1) # [1, 32, 1]

        self.n=torch.arange(start=0,end=self.f_dim)
        self.n=self.n.unsqueeze(0).unsqueeze(0)  #[1,1,F]

        table = 2 * np.pi * self.n * freqs

        #print(freqs.shape, x.shape, table.shape)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) #[1,32,F]

        return table
    

    def forward(self, input_tensor):

        #print(input_tensor.shape)
        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[-1]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.embeddings, [batch_size_tensor, time_dim,self.N*2, self.f_dim])
        fembeddings_2=fembeddings_2.permute(0,2,3,1)
    
        
        #print(input_tensor.shape, fembeddings_2.shape)
        return torch.cat((input_tensor,fembeddings_2),1)  


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, num_buckets: int, max_distance: int
    ):
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int):
        i, j, device = num_queries, num_keys, self.relative_attention_bias.weight.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = einops.rearrange(k_pos, "j -> 1 j") - einops.rearrange(q_pos, "i -> i 1")

        relative_position_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        bias = self.relative_attention_bias(relative_position_bucket)
        bias = einops.rearrange(bias, "m n h -> 1 h m n")
        return bias

class TimeAttentionBlock(nn.Module):
    def __init__(self, Nin,attention_dict, init, init_zero, Fdim) -> None:
        super().__init__()
        #NA=attention_dict.N
        self.attention_dict=attention_dict
        self.Fdim=Fdim
        N=attention_dict.num_heads*Fdim 
        self.qk = Conv1d(N, N*2, bias=self.attention_dict.bias_qkv, **init )
        self.proj_in=Conv2d(Nin, attention_dict.num_heads, (1,1), bias=False, **init)
        self.proj_out=Conv2d(attention_dict.num_heads, Nin, (1,1), bias=False, **init)
        #not sure if a bias is a good idea here
        #self.v = Conv2d(N, N*2, (1,1), bias=False,**init )
        #I think that as long as the main signal path layers are bias free, we should be safe from artifacts
        #self.proj = Conv1d(NA, NA, 1, bias=False, **init)

        self.scale=(N/self.attention_dict.num_heads)**-0.5
        self.use_rel_pos = self.attention_dict.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos = RelativePositionBias(
                num_buckets=attention_dict.rel_pos_num_buckets,
                max_distance=attention_dict.rel_pos_max_distance,
                num_heads=attention_dict.num_heads,
            )

    def forward(self, x):
        #shape of x is [batch, C,F, T]

        #we need shape: [batch, heads, T, D]
        #with heands on different (original) channels
        #print(x.shape, self.Fdim)

        x=self.proj_in(x) #reduce the C dimensionality

        #print(x.shape, self.Fdim)
        #normalize everyting (easy)

        #split into heads
        x=einops.rearrange(x, "b h f t -> b (h f) t")

        v=einops.rearrange(x,"b (h f) t -> b h t f", f=self.Fdim) #identity layer for the values

        qk=self.qk(x) #linear layer
        #for now, f are features (all merged) but still represents frequency

        qk=einops.rearrange(qk, "b (h d) t -> b h t d", h=self.attention_dict.num_heads)
        q,k=qk.chunk(2,dim=-1)

        #print("qk",q.shape, k.shape)
        sim = torch.einsum("... n d, ... m d -> ... n m", q, k)
        #print("sim",sim.shape)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        #print("sim",sim.shape)
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        #print("attn",attn.shape, v.shape)
        out = torch.einsum("... n m, ... m d -> ... n d", attn, v)

        #print("out",out.shape)
        out = einops.rearrange(out, "b h t f -> b h f t", f=self.Fdim)
        #out = einops.rearrange(out, "b (h f) t -> b h f t", f=self.Fdim)

        #reverse step
        out=self.proj_out(out)

        return out
        
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=True,
        num_dils = 6,
        bias=False,
        kernel_size=(5,3),
        emb_dim=512,
        proj_place='before', #using 'after' in the decoder out blocks
        init=None,
        init_zero=None,
        attention_dict=None,
        Fdim=128, #number of frequency bins
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.num_dils=num_dils
        self.proj_place=proj_place
        self.Fdim=Fdim

        if self.proj_place=='before':
            #dim_out is the block dimension
            N=dim_out
        else:
            #dim in is the block dimension
            N=dim
            self.proj_out = Conv2d(N, dim_out,   bias=bias, **init) if N!=dim_out else nn.Identity() #linear projection

        self.res_conv = Conv2d(dim, dim_out, bias=bias, **init) if dim!= dim_out else nn.Identity() #linear projection
        self.proj_in = Conv2d(dim, N,   bias=bias, **init) if dim!=N else nn.Identity()#linear projection



        self.H=nn.ModuleList()
        self.affine=nn.ModuleList()
        self.gate=nn.ModuleList()
        if self.use_norm:
            self.norm=nn.ModuleList()

        for i in range(self.num_dils):

            if self.use_norm:
                self.norm.append(BiasFreeGroupNorm(N,8))

            self.affine.append(Linear(emb_dim, N, **init))
            self.gate.append(Linear(emb_dim, N, **init_zero))
            #self.H.append(Gated_residual_layer(dim_out, (5,3), (2**i,1), bias=bias)) #sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)
            self.H.append(Conv2d(N,N,    
                                    kernel=kernel_size,
                                    dilation=(2**i,1),
                                    bias=bias, **init)) #freq convolution (dilated) 

        self.attention_dict=attention_dict
        if self.attention_dict is not None:
            #NA=self.attention_dict.N
            self.norm2=BiasFreeGroupNorm(N,8)
            self.affine2=Linear(emb_dim, N, **init)
            self.gate2=Linear(emb_dim, N, **init_zero)
            #self.norm2 = BiasFreeGroupNorm(N,8)
            #self.proj_attn_in = Conv1d(N*Fdim, NA,   bias=bias, **init) if (N*Fdim)!=NA else nn.Identity()#linear projection
            #self.proj_attn_out = Conv1d(NA, N*Fdim,   bias=bias, **init_zero) if NA!=(N*Fdim) else nn.Identity() #linear projection
            ##the attention is applied time-wise, since channels times frequency is too much, we need to reduce the dimensionality using a linear projection
            self.attn_block=TimeAttentionBlock(N,self.attention_dict, init,init_zero, self.Fdim)



    def forward(self, input_x, sigma):
        
        x=input_x

        x=self.proj_in(x)

        if self.attention_dict is not None:
            i_x=x

            gamma=self.affine2(sigma)
            scale=self.gate2(sigma)

            x=self.norm2(x)
            x=x*(gamma.unsqueeze(2).unsqueeze(3)+1) #no bias

            x=self.attn_block(x)*scale.unsqueeze(2).unsqueeze(3)

            #x=(x+i_x)
            x=(x+i_x)/(2**0.5)

        for norm, affine, gate, conv in zip(self.norm, self.affine, self.gate, self.H):
            x0=x
            if self.use_norm:
                x=norm(x)
            gamma =affine(sigma)
            scale=gate(sigma)

            x=x*(gamma.unsqueeze(2).unsqueeze(3)+1) #no bias


            x=(x0+conv(F.gelu(x))*scale.unsqueeze(2).unsqueeze(3))/(2**0.5) 
            #x=(x0+conv(F.gelu(x))*scale.unsqueeze(2).unsqueeze(3))
        
        #one residual connection here after the dilated convolutions


        if self.proj_place=='after':
            x=self.proj_out(x)

        x=(x + self.res_conv(input_x))/(2**0.5)

        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for inpainting: allows the gap region to attend to the context encoding.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gap_feat, context_feat, gap_mask=None, context_mask=None):
        # gap_feat: (B, L_gap, D)
        # context_feat: (B, L_ctx, D)
        # gap_mask/context_mask: (B, L_gap/L_ctx) (optional)
        attn_output, _ = self.attn(
            query=gap_feat,
            key=context_feat,
            value=context_feat,
            key_padding_mask=context_mask
        )
        return self.norm(gap_feat + attn_output)

class AttentionOp(torch.autograd.Function):

    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic': 
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3': 
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
class UpDownResample(nn.Module):
    def __init__(self,
        up=False, 
        down=False,
        mode_resample="T", #T for time, F for freq, TF for both
        resample_filter='cubic', 
        pad_mode='reflect'
        ):
        super().__init__()
        assert not (up and down) #you cannot upsample and downsample at the same time
        assert up or down #you must upsample or downsample
        self.down=down
        self.up=up
        if up or down:
            #upsample block
            self.pad_mode = pad_mode #I think reflect is a goof choice for padding
            self.mode_resample=mode_resample
            if mode_resample=="T":
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            elif mode_resample=="F":
                #kerel shouuld be the same
                kernel_1d = torch.tensor(_kernels[resample_filter], dtype=torch.float32)
            else:
                raise NotImplementedError("Only time upsampling is implemented")
                #TODO implement freq upsampling and downsampling
            self.pad = kernel_1d.shape[0] // 2 - 1
            self.register_buffer('kernel', kernel_1d)
    def forward(self, x):
        shapeorig=x.shape
        #x=x.view(x.shape[0],-1,x.shape[-1])
        x=x.view(-1,x.shape[-2],x.shape[-1]) #I have the feeling the reshape makes everything consume too much memory. There is no need to have the channel dimension different than 1. I leave it like this because otherwise it requires a contiguous() call, but I should check if the memory gain / speed, would be significant.
        if self.mode_resample=="F":
            x=x.permute(0,2,1)#call contiguous() here?

        #print("after view",x.shape)
        if self.down:
            x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        elif self.up:
            x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)

        #print("after pad",x.shape)

        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        #print("weight",weight.shape)
        indices = torch.arange(x.shape[1], device=x.device)
        #print("indices",indices.shape)
        #weight = self.kernel.to(x.device).unsqueeze(0).unsqueeze(0).expand(x.shape[1], x.shape[1], -1)
        #print("weight",weight.shape)
        weight[indices, indices] = self.kernel.to(weight)
        if self.down:
            x_out= F.conv1d(x, weight, stride=2)
        elif self.up:
            x_out =F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)

        if self.mode_resample=="F":
            x_out=x_out.permute(0,2,1).contiguous()
            return x_out.view(shapeorig[0],-1,x_out.shape[-2], shapeorig[-1])
        else:
            return x_out.view(shapeorig[0],-1,shapeorig[2], x_out.shape[-1])


class Unet_CQT_oct_with_attention(nn.Module):
    """
    U-Net with octave CQT encoder/decoder and a cross-attention bottleneck
    for audio in-painting.  All tensor shapes are documented inline.
    """
    def __init__(self, args, device):
        super().__init__()
        args = EasyDict(args)
        self.args   = args
        self.device = device

        # ---  hyper-params ---------------------------------------------------
        self.emb_dim     = args.network.emb_dim
        self.bins_per_oct= args.network.cqt.bins_per_oct
        self.num_octs    = args.network.cqt.num_octs
        self.fbins       = self.bins_per_oct * self.num_octs

        # ------------ utilities ---------------------------------------------
        init      = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=1e-7)

        # time-step embedding
        self.embedding = RFF_MLP_Block(emb_dim=self.emb_dim, init=init)

        # CQT
        self.CQTransform = CQT_nsgt(
            numocts   = self.num_octs,
            binsoct   = self.bins_per_oct,
            mode      = "oct",
            fs        = self.args.exp.sample_rate,
            audio_len = self.args.exp.audio_len,
            device    = self.device,
        )

        # optional frequency positional encoding for each octave
        self.use_fencoding = args.network.use_fencoding
        if self.use_fencoding:
            N_freq_encoding = 32
            self.freq_encodings = nn.ModuleList([
                AddFreqEncodingRFF(self.bins_per_oct, N_freq_encoding)
                for _ in range(self.num_octs)
            ])
            Nin = 2 * N_freq_encoding + 2
        else:
            Nin = 2  # real+imag channels only

        # ---------------- encoder / decoder parameter lists ------------------
        self.Ns          = args.network.Ns          # channels per octave
        self.num_dils    = args.network.num_dils    # dilations per octave
        self.attn_layers = args.network.attention_layers
        self.attn_dict   = args.network.attention_dict
        self.use_norm    = args.network.use_norm

        self.downs      = nn.ModuleList()
        self.middle     = nn.ModuleList()
        self.ups        = nn.ModuleList()

        self.downsamplerT = UpDownResample(down=True,  mode_resample="T")
        self.upsamplerT   = UpDownResample(up=True,    mode_resample="T")

        # ---------- down path ------------------------------------------------
        for i in range(self.num_octs):
            dim_in  = self.Ns[i-1] if i else self.Ns[i]
            dim_out = self.Ns[i]
            attn    = self.attn_dict if self.attn_layers[i] else None

            self.downs.append(nn.ModuleList([
                ResnetBlock(Nin,      dim_in,  self.use_norm,
                            num_dils=1, bias=False, kernel_size=(1,1),
                            emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                Conv2d(2, dim_out, kernel=(5,3), bias=False, **init),
                ResnetBlock(dim_in, dim_out, self.use_norm,
                            num_dils=self.num_dils[i], bias=False,
                            attention_dict=attn, emb_dim=self.emb_dim,
                            init=init, init_zero=init_zero,
                            Fdim=(i+1)*self.bins_per_oct)
            ]))

        # ---------- bottleneck ----------------------------------------------
        if args.network.bottleneck_type != "res_dil_convs":
            raise NotImplementedError

        for _ in range(args.network.num_bottleneck_layers):
            attn = self.attn_dict if self.attn_layers[-1] else None
            self.middle.append(nn.ModuleList([
                ResnetBlock(self.Ns[-1], 2, self.use_norm, num_dils=1,
                            bias=False, kernel_size=(1,1), proj_place="after",
                            emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                ResnetBlock(self.Ns[-1], self.Ns[-1], self.use_norm,
                            num_dils=self.num_dils[-1], bias=False,
                            attention_dict=attn, emb_dim=self.emb_dim,
                            init=init, init_zero=init_zero,
                            Fdim=self.num_octs*self.bins_per_oct)
            ]))

        # ---------- up path --------------------------------------------------
        for i in range(self.num_octs-1, -1, -1):
            dim_in  = self.Ns[i]*2
            dim_out = self.Ns[i-1] if i else self.Ns[0]
            attn    = self.attn_dict if self.attn_layers[i] else None

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out, 2, self.use_norm, num_dils=1,
                            bias=False, kernel_size=(1,1), proj_place="after",
                            emb_dim=self.emb_dim, init=init, init_zero=init_zero),
                ResnetBlock(dim_in,  dim_out, self.use_norm,
                            num_dils=self.num_dils[i], bias=False,
                            attention_dict=attn, emb_dim=self.emb_dim,
                            init=init, init_zero=init_zero,
                            Fdim=(i+1)*self.bins_per_oct)
            ]))

        # ---------- cross-attention in bottleneck ---------------------------
        self.cross_attn   = CrossAttentionBlock(d_model=self.emb_dim, n_heads=4)
        self.gap2emb      = nn.Linear(self.bins_per_oct, self.emb_dim)
        self.emb2gap      = nn.Linear(self.emb_dim,      self.bins_per_oct)

    # ------------------------------------------------------------------------
    def forward(self, audio_noisy, context, mask, sigma):
        """
        B  – batch, T  – time samples, F  – freq bins per octave
        """
        B, T = audio_noisy.shape

        # ---------- noise scale embedding -----------------------------------
        sigma_emb = self.embedding(sigma)

        # ---------- CQT ------------------------------------------------------
        X_list      = self.CQTransform.fwd(audio_noisy.unsqueeze(1))
        X_list_ctx  = self.CQTransform.fwd(context    .unsqueeze(1))

        # deepest octave ⇒ bottleneck feature
        X_bott      = X_list     [-1].squeeze(1)       # (B, F, Tb)
        X_ctx_bott  = X_list_ctx [-1].squeeze(1)       # (B, F, Tb)
        X_bott      = X_bott.permute(0,2,1)            # (B, Tb, F)
        X_ctx_bott  = X_ctx_bott.permute(0,2,1)        # (B, Tb, F)

        # ---------- resample mask to Tb -------------------------------------
        mask_ds     = F.interpolate(mask.unsqueeze(1).float(),
                                    size=X_bott.shape[1], mode="nearest"
                                   ).squeeze(1)        # (B, Tb)
        gap_mask    = mask_ds == 0
        ctx_mask    = mask_ds == 1

        # ---------- build gap / ctx sequences -------------------------------
        gap_feats, ctx_feats = [], []
        for b in range(B):
            gap_idx = gap_mask[b].nonzero(as_tuple=True)[0]
            ctx_idx = ctx_mask[b].nonzero(as_tuple=True)[0]
            gap_feats.append(X_bott     [b, gap_idx])  # (Ngap, F)
            ctx_feats.append(X_ctx_bott [b, ctx_idx])  # (Nctx, F)

        max_gap = max(g.shape[0] for g in gap_feats)
        max_ctx = max(c.shape[0] for c in ctx_feats)
        gap_feats = [F.pad(g, (0,0,0,max_gap-g.shape[0])) for g in gap_feats]
        ctx_feats = [F.pad(c, (0,0,0,max_ctx-c.shape[0])) for c in ctx_feats]
        gap_feats = torch.stack(gap_feats, 0)          # (B, Ngap*, F)
        ctx_feats = torch.stack(ctx_feats, 0)          # (B, Nctx*, F)

        # --- build real features for attention ------------------------------------
        gap_feats_real = gap_feats.real       # take only the real part
        ctx_feats_real = ctx_feats.real

        gap_feats_real = self.gap2emb(gap_feats_real)
        ctx_feats_real = self.gap2emb(ctx_feats_real)

        gap_feats_real = self.cross_attn(gap_feats_real, ctx_feats_real)
        gap_feats_real = self.emb2gap(gap_feats_real)

        # convert back to complex with zero-imag part
        gap_feats = torch.complex(gap_feats_real, torch.zeros_like(gap_feats_real))

        # ---------- scatter back into bottleneck ----------------------------
        X_bott_attn = X_bott.clone()                   # (B, Tb, F)
        for b in range(B):
            gap_idx = gap_mask[b].nonzero(as_tuple=True)[0]
            X_bott_attn[b, gap_idx] = gap_feats[b, :gap_idx.numel()]
        X_list      = list(X_list)                     # make it mutable
        X_list[-1]  = X_bott_attn.permute(0,2,1).unsqueeze(1)

        # --------------------------------------------------------------------
        hs = []
        X  = None
        pyr= None
        sigma_emb = self.embedding(sigma) 
        # ---------------- encoder -------------------------------------------
        for i, (init_block, pyr_down_proj, res_block) in enumerate(self.downs):
            C   = X_list[-1-i].squeeze(1)              # (B, F_i, T_i)
            C   = torch.view_as_real(C)                # (B, F_i, T_i, 2)
            C   = C.permute(0,3,1,2).contiguous()      # (B, 2, F_i, T_i)
            C2  = self.freq_encodings[i](C) if self.use_fencoding else C
            C2  = init_block(C2, sigma_emb)

            if i == 0:
                X   = C2
                pyr = self.downsamplerT(C)
            elif i < self.num_octs - 1:
                pyr = torch.cat([self.downsamplerT(C), self.downsamplerT(pyr)],
                                dim=2)
                X   = torch.cat([C2, X], dim=2)
            else:  # last octave
                pyr = torch.cat([C, pyr], dim=2)
                X   = torch.cat([C2, X], dim=2)

            X   = res_block(X, sigma_emb)

            # downsample main path except last octave
            if i < self.num_octs - 1:
                X = self.downsamplerT(X)

            # residual combine
            X = (X + pyr_down_proj(pyr)) / np.sqrt(2)
            hs.append(X)

        # ---------------- bottleneck layers ---------------------------------
        for out_block, res_block in self.middle:
            X    = res_block(X, sigma_emb)
            Xout = out_block(X, sigma_emb)                 # first assignment

        def _match_size(t1, t2):
            """
            Center-crop the larger tensor so that t1 and t2 have identical
            (F, T) shapes.  Both tensors are (B,C,F,T).
            """
            _, _, F1, T1 = t1.shape
            _, _, F2, T2 = t2.shape
            Fmin, Tmin = min(F1, F2), min(T1, T2)
            if F1 != Fmin or T1 != Tmin:
                dF = (F1 - Fmin) // 2
                dT = (T1 - Tmin) // 2
                t1 = t1[:, :, dF:dF+Fmin, dT:dT+Tmin]
            if F2 != Fmin or T2 != Tmin:
                dF = (F2 - Fmin) // 2
                dT = (T2 - Tmin) // 2
                t2 = t2[:, :, dF:dF+Fmin, dT:dT+Tmin]
            return t1, t2

        # ---------------- decoder -------------------------------------------
        X_list_out = [None] * self.num_octs
        for i, (out_block, res_block) in enumerate(self.ups):
            j   = len(self.ups) - i - 1
            skip= hs.pop()
            X, skip = _match_size(X, skip)
            X   = torch.cat([X, skip], dim=1)
            X   = res_block(X, sigma_emb)

            out_new = out_block(X, sigma_emb)          # fresh prediction
            # ---> ensure Xout and out_new match in (F,T) <---
            Xout, out_new = _match_size(Xout, out_new) #  <<< add this line
            Xout = (Xout + out_new) / np.sqrt(2)

            if j <= self.num_octs - 1:
                # split frequency axis
                X      =  X[:,:, self.bins_per_oct: , :]
                Out, Xout = Xout[:,:, :self.bins_per_oct , :], Xout[:,:, self.bins_per_oct: , :]

                Out  = Out.permute(0,2,3,1).contiguous()
                Out  = torch.view_as_complex(Out)
                X_list_out[j] = Out.unsqueeze(1)

            if 0 < j <= self.num_octs - 1:
                X    = self.upsamplerT(X)
                Xout = self.upsamplerT(Xout)
                
        # ---------------- time-axis fix before inverse CQT ------------------------
        def _match_time(x, target_len):
            """
            Center-crop or zero-pad complex tensor `x`
            so that x.shape[-1] == target_len.
            x has shape (B,1,F,T), dtype complex.
            """
            T = x.shape[-1]
            if T == target_len:
                return x
            if T > target_len:                       # crop
                trim = (T - target_len) // 2
                return x[..., trim:trim+target_len]
            else:                                    # pad
                pad = target_len - T
                left = pad // 2
                right = pad - left
                return F.pad(x, (left, right))       # PyTorch pads complex tensors

        # make sure every octave entry exists and has correct T
        for i in range(self.num_octs):
            if X_list_out[i] is None:                # safety: missing octave
                X_list_out[i] = torch.zeros_like(X_list[i])
            X_list_out[i] = _match_time(
                X_list_out[i],
                X_list[i].shape[-1]                  # original CQT frame count
            )

        # ---------------- inverse CQT ---------------------------------------
        pred_time = self.CQTransform.bwd(X_list_out).squeeze(1)
        pred_time = pred_time[:, :T]                   # crop to original length
        assert pred_time.shape == audio_noisy.shape
        return pred_time
