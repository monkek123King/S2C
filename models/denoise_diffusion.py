# https://github.com/lucidrains/denoising-diffusion-pytorch
import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce

from tqdm.auto import tqdm
# from mamba_ssm import Mamba
from models.diff_multiAttention import MultiheadFusionAttn
from models.utils import init_biased_mask, enc_dec_mask
# constants

import copy

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x_self, x_other):
        """
        x_self: (B, D, C) - feature of self hand        (16. 2048, 30)
        x_other: (B, D, C) - feature of other hand      (16. 2048, 30)
        return: (B, D, C) - fused self hand             (16. 2048, 30)
        """

        x_self = x_self.permute(0, 2, 1)  # (16. 30, 2048)
        x_other = x_other.permute(0, 2, 1)  # (16. 30, 2048)

        # 交叉注意力计算
        x_fused, _ = self.cross_attn(x_self, x_other, x_other)  # (16. 30, 2048)

        # 残差连接 + 归一化
        x_fused = self.norm(x_fused + x_self)  # (16. 30, 2048)

        # MLP 进一步处理
        x_fused = self.mlp(x_fused)  # (16. 30, 2048)

        # 维度转换
        x_fused = x_fused.permute(0, 2, 1)  # (16. 30, 2048)

        return x_fused


# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        condition=True,
        condition_dim=512,
        guide=True,
        guide_dim=3,
        self_condition=False,
        encoder_type='none',
        num_layer=16,
        resnet_block_groups = 8,
        remap_noise=True,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        feature_fusion = 0,
    ):
        super().__init__()

        # determine dimensions

        self.x_cond = None
        self.channels = channels
        self.self_condition = self_condition

        self.condition = condition
        if condition:
            self.remap_noise = remap_noise
            if self.remap_noise:
                self.cond_feature_map = nn.Linear(channels, condition_dim)
                input_channels = condition_dim + condition_dim
            else:
                input_channels = channels + condition_dim
        else:
            input_channels = channels * (2 if self_condition else 1)
        if guide:
            self.guide = True
            self.guide_dim = guide_dim
        else:
            self.guide = False
            self.guide_dim = 0

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels + self.guide_dim, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]

        self.feature_fusion = feature_fusion      # 0: concat -> self_attn 1: self_attn -> concat, 2: cross_attn -> self_attn, 3: self_attn -> cross_attn 4:diff atten

        print(f'!!! Use feature_fusion way: {self.feature_fusion}!!!')

        if self.feature_fusion == 0:
            self.mid_block1 = block_klass(mid_dim + mid_dim, mid_dim, time_emb_dim = time_dim)
        else:
            self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
            self.cross_attn = CrossAttentionFusion(embed_dim=mid_dim) 

        if self.feature_fusion == 4:
            self.diff_attn = MultiheadFusionAttn(embed_dim=mid_dim, num_heads=8,depth=12)
            
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))

        if self.feature_fusion == 1:
            self.mid_block2 = block_klass(mid_dim + mid_dim, mid_dim, time_emb_dim = time_dim)
        else:
            self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)


        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

        self.encoder_type = encoder_type
        self.feature_dim = condition_dim
        if self.encoder_type == 'transformer':
            print('!!! Use transformer in denoising Unet !!!')
            self.biased_mask1 = init_biased_mask(n_head=4, max_seq_len=900, period=30)

            decoder_layer = nn.TransformerDecoderLayer(d_model=self.feature_dim, nhead=4, dim_feedforward=self.feature_dim,
                                                       batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=4, batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layer)
        elif self.encoder_type == 'mamba':
            print('!!! Use mamba in denoising Unet !!!')
            # self.mamba_encoder = Mamba(
            #                         d_model=self.feature_dim, # Model dimension d_model
            #                         d_state=num_layer,  # SSM state expansion factor
            #                         d_conv=4,    # Local convolution width
            #                         expand=2,    # Block expansion factor
            #                     ).to("cuda")

    def set_xcond(self, x_cond):
        if self.encoder_type == 'transformer':
            ## get mask
            tgt_mask = None
            if x_cond.shape[0] == 1:
                tgt_mask = self.biased_mask1[:, :x_cond.shape[2],
                           :x_cond.shape[2]].clone().detach().to(device=x_cond.device)
            memory_mask = enc_dec_mask(x_cond.device, x_cond.shape[2], x_cond.shape[2])

            time_queries = x_cond.clone().permute(0, 2, 1)

            x_cond = self.transformer_decoder(time_queries, self.transformer_encoder(x_cond.permute(0, 2, 1)),
                                              tgt_mask=tgt_mask, memory_mask=memory_mask).permute(0, 2, 1)
        # elif self.encoder_type == 'mamba':
        #     x_cond = self.mamba_encoder(x_cond.permute(0, 2, 1)).permute(0, 2, 1)
        self.x_cond = x_cond

    def forward_down(self, x, time, x_self_cond = None, x_cond=None, guide=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        if self.condition:
            if self.remap_noise:
                x = self.cond_feature_map(x.permute(0, 2, 1)).permute(0, 2, 1)
                x_cond = self.x_cond
            x = torch.cat((x_cond, x), dim = 1)

        if self.guide:
            x = torch.cat([x, guide], 1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        if self.feature_fusion == 1:
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)  
        elif self.feature_fusion == 3 or self.feature_fusion == 4:
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t) 

        return x, r, t, h
    
    def forward(self, x_self, x_other, r, t, h):
        if self.feature_fusion == 0:
            x = torch.cat((x_self, x_other), dim = 1)
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)
        elif self.feature_fusion == 1:
            x = torch.cat((x_self, x_other), dim = 1)
            x = self.mid_block2(x, t)
        elif self.feature_fusion == 2:
            x = self.cross_attn(x_self, x_other)
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)
        elif self.feature_fusion == 3:       
            x = self.cross_attn(x_self, x_other)
        elif self.feature_fusion == 4:
            x = self.diff_attn(x_self, x_other)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D_piano2pose(nn.Module):
    def __init__(
        self,
        model,
        piano2posi,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        train_acceLoss_weight = 0.1,
        train_velLoss_weight = 0.1,
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective
        self.train_acceLoss_weight = train_acceLoss_weight
        self.train_velLoss_weight = train_velLoss_weight
        print(f'!!! Use train_acceLoss_weight: {self.train_acceLoss_weight} and train_velLoss_weight: {self.train_velLoss_weight} !!!')

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.piano2posi = piano2posi
        self.piano2posi.requires_grad_(False)
        # self.audio_encoder_cont.feature_extractor._freeze_parameters()

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_self, x_other, t, x_self_cond_self=None, x_self_cond_other=None, cond=None, guide=None, cond_other=None, guide_other=None, model_other=None, clip_x_start = False, rederive_pred_noise = False):
        # model_output = self.model(x, t, x_self_cond, cond, guide)
        x_down_self, r_self, t_emb_self, h_self = self.model.forward_down(x_self, t, x_self_cond_self, cond, guide)
        x_down_other, r_other, t_emb_other, h_other = model_other.forward_down(x_other, t, x_self_cond_other, cond_other, guide_other)
        model_output_self = self.model(x_down_self, x_down_other, r_self, t_emb_self, h_self)
        model_output_other = model_other.forward(x_down_other, x_down_self, r_other, t_emb_other, h_other)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise_self = model_output_self
            x_start_self = self.predict_start_from_noise(x_self, t, pred_noise_self)
            x_start_self = maybe_clip(x_start_self)

            pred_noise_other = model_output_other
            x_start_other = self.predict_start_from_noise(x_other, t, pred_noise_other)
            x_start_other = maybe_clip(x_start_other)

            if clip_x_start and rederive_pred_noise:
                pred_noise_self = self.predict_noise_from_start(x_self, t, x_start_self)
                pred_noise_other = self.predict_noise_from_start(x_other, t, x_start_other)

        elif self.objective == 'pred_x0':
            x_start_self = model_output_self
            x_start_other = model_output_other
            x_start_self = maybe_clip(x_start_self)
            x_start_other = maybe_clip(x_start_other)
            pred_noise_self = self.predict_noise_from_start(x_self, t, x_start_self)
            pred_noise_other = self.predict_noise_from_start(x_other, t, x_start_other)

        elif self.objective == 'pred_v':
            v_self = model_output_self
            v_other = model_output_other
            x_start_self = self.predict_start_from_v(x_self, t, v_self)
            x_start_other = self.predict_start_from_v(x_other, t, v_other)
            x_start_self = maybe_clip(x_start_self)
            x_start_other = maybe_clip(x_start_other)
            pred_noise_self = self.predict_noise_from_start(x_self, t, x_start_self)
            pred_noise_other = self.predict_noise_from_start(x_other, t, x_start_other)

        return ModelPrediction(pred_noise_self, x_start_self), ModelPrediction(pred_noise_other, x_start_other)

    def p_mean_variance(self, x_self, x_other, t, x_self_cond_self=None, x_self_cond_other=None, cond=None, guide=None, cond_other=None, guide_other=None, model_other=None, clip_denoised = True):
        # preds = self.model_predictions(x, t, x_self_cond, cond, guide)
        preds_self, preds_other = self.model_predictions(x_self, x_other, t, x_self_cond_self, x_self_cond_other, cond, guide, cond_other, guide_other, model_other)
        x_start_self = preds_self.pred_x_start
        x_start_other = preds_other.pred_x_start

        if clip_denoised:
            x_start_self.clamp_(-1., 1.)
            x_start_other.clamp_(-1., 1.)

        model_mean_self, posterior_variance_self, posterior_log_variance_self = self.q_posterior(x_start = x_start_self, x_t = x_self, t = t)
        model_mean_other, posterior_variance_other, posterior_log_variance_other = self.q_posterior(x_start = x_start_other, x_t = x_other, t = t)
        return model_mean_self, posterior_variance_self, posterior_log_variance_self, x_start_self, model_mean_other, posterior_variance_other, posterior_log_variance_other, x_start_other

    @torch.no_grad()
    def p_sample(self, x_self, x_other, t: int, x_self_cond_self=None, x_self_cond_other=None, cond=None, guide=None, cond_other=None, guide_other=None, model_other=None, clip_denoised = True):
        b, *_, device = *x_self.shape, x_self.device
        batched_times = torch.full((b,), t, device = x_self.device, dtype = torch.long)
        model_mean_self, _, model_log_variance_self, x_start_self, model_mean_other, _, model_log_variance_other, x_start_other = self.p_mean_variance(x_self, x_other, batched_times, x_self_cond_self, x_self_cond_other, cond, guide, cond_other, guide_other, model_other, clip_denoised)
        noise_self = torch.randn_like(x_self) if t > 0 else 0. # no noise if t == 0
        noise_other = torch.randn_like(x_other) if t > 0 else 0. # no noise if t == 0
        pred_img_self = model_mean_self + (0.5 * model_log_variance_self).exp() * noise_self
        pred_img_other = model_mean_other + (0.5 * model_log_variance_other).exp() * noise_other
        return pred_img_self, x_start_self, pred_img_other, x_start_other

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, guide, cond_other, guide_other, model_other):
        batch, device = shape[0], self.betas.device

        img_self = torch.randn(shape, device=device)
        # img_other = torch.randn(shape, device=device)
        img_other = copy.deepcopy(img_self)

        x_start_self = None
        x_start_other = None
        self.model.set_xcond(cond)
        model_other.set_xcond(cond_other)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond_self = x_start_self if self.self_condition else None
            self_cond_other = x_start_other if self.self_condition else None
            img_self, x_start_self, img_other, x_start_other = self.p_sample(img_self, img_other, t, self_cond_self, self_cond_other, cond, guide, cond_other, guide_other, model_other)

        # img = self.unnormalize(img)
        return img_self, img_other

    @torch.no_grad()
    def ddim_sample(self, shape, cond, guide, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, cond, guide, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, guide, cond, guide_other, cond_other, model_other, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        guide = guide.permute(0, 2, 1)
        guide_other = guide_other.permute(0, 2, 1)
        cond = cond.permute(0, 2, 1)
        cond_other = cond_other.permute(0, 2, 1)
        return sample_fn((batch_size, channels, seq_length), cond, guide, cond_other, guide_other, model_other)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def calculate_velocity(self, x_start):
        velocity = x_start[:, :, 1:] - x_start[:, :, :-1]  
        return velocity

    def calculate_acceleration(self, x_start):
        velocity = self.calculate_velocity(x_start)
        acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]
        return acceleration
    
    def p_losses(self, x_down_self, x_down_other, r, t, t_emb, h, noise, x_start, x_t):
        model_out = self.model(x_down_self, x_down_other, r, t_emb, h)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
            start_predict = model_out
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            start_predict = self.predict_start_from_v(x_t, t, model_out)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        
        velocity_weight = self.train_velLoss_weight  # 速度损失的权重
        acceleration_weight = self.train_acceLoss_weight  # 加速度损失的权重
       
        if velocity_weight > 0:
            gt_velocity = self.calculate_velocity(x_start)  
            out_velocity = self.calculate_velocity(start_predict) 
            velocity_loss = F.mse_loss(out_velocity, gt_velocity, reduction='none')
            velocity_loss = reduce(velocity_loss, 'b ... -> b', 'mean')
            loss += velocity_weight * velocity_loss  
        if acceleration_weight > 0:
            gt_acceleration = self.calculate_acceleration(x_start)  
            out_acceleration = self.calculate_acceleration(start_predict) 
            acceleration_loss = F.mse_loss(out_acceleration, gt_acceleration, reduction='none')
            acceleration_loss = reduce(acceleration_loss, 'b ... -> b', 'mean')
            loss += acceleration_weight * acceleration_loss  
        
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, x_self, x_other, r, t, t_emb, h, noise, x_start, x_t):
        return self.p_losses(x_self, x_other, r, t, t_emb, h, noise, x_start, x_t)
    
    def forward_down(self, img, t, guide, cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(img))

        guide = guide.permute(0, 2, 1).detach()
        cond = cond.permute(0, 2, 1).detach()
        x = self.q_sample(x_start = img, t = t, noise = noise)

        x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t, cond=cond, guide=guide).pred_x_start
        #         x_self_cond.detach_()

        self.model.set_xcond(cond)
        return self.model.forward_down(x, t, x_self_cond, cond, guide), noise, x
    
    def getGuideAndCond(self, img, audio):
        b, c, n = img.shape
        with torch.no_grad():
            guide, cond = self.piano2posi(audio, n, return_hidden=True)   
        return guide, cond