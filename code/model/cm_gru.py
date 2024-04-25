from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,     # RBS up
    ResidualBlockWithStride,   # RBS down
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def ste_round(x: Tensor) -> Tensor:  # ste_round(x)的梯度==x的梯度
    return torch.round(x) - x.detach() + x

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride) 

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


class ConvGRU(nn.Module):
    """  GRUCell designed for entropy model  只完成一个时间步的计算
    """
    def __init__(self, hidden_dim, input_dim):
        super(ConvGRU, self).__init__()
        self.conv_z_x, self.conv_z_h = conv3x3(input_dim, hidden_dim), conv3x3(hidden_dim, hidden_dim)
        self.conv_r_x, self.conv_r_h = conv3x3(input_dim, hidden_dim), conv3x3(hidden_dim, hidden_dim)
        self.conv_h_x, self.conv_h_u = conv3x3(input_dim, hidden_dim), conv3x3(hidden_dim, hidden_dim)
        # self.conv_out = conv3x3(hidden_dim, hidden_dim)


    def forward(self, h, x):
        """
          x : [batch_size, c, h, w]   当前输入特征
          h : [batch_size, hidden, h, w]   隐特征
          return  h 更新后的隐特征
        """
        z = torch.sigmoid(self.conv_z_x(x) + self.conv_z_h(h))
        r = torch.sigmoid(self.conv_r_x(x) + self.conv_r_h(h))
        h_hat = torch.tanh(self.conv_h_x(x) + self.conv_h_u(torch.mul(r, h)))
        h = torch.mul((1. - z), h) + torch.mul(z, h_hat)
        # y = self.conv_out(h)
        # return y, h

        return h
    

class mySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class CA(nn.Module):
    """ cross attention model
    """

    def __init__(self, input_dim, output_dim, head_dim, k_num=32, text_dim=512):
        super(CA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.text_dim = text_dim
        self.head_dim = head_dim
        self.k_num = k_num
        self.scale = self.head_dim ** -0.5
        self.n_heads = self.input_dim//self.head_dim

        self.embedding_q = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.embedding_k = nn.Linear(self.input_dim // 2, self.input_dim * self.k_num, bias=True)
        self.embedding_v = nn.Linear(self.input_dim // 2, self.input_dim * self.k_num, bias=True)
        self.embedding_muti_head = nn.Sequential(
            nn.Linear(self.input_dim, 4 * self.input_dim),
            nn.GELU(),
            nn.Linear(4 * self.input_dim, self.output_dim),
        )

    def forward(self, x, y):
       """  
       Args:
           x: input tensor with shape of [b h w c]    (q)
           y: input tensor with shape of [b Len]   (k, v)
       """
       h_size = x.shape[1]
       q, k, v = self.embedding_q(x), self.embedding_k(y), self.embedding_v(y)
       q = rearrange(q, 'b h w (head c) -> head b (h w) c', c=self.head_dim)
       k, v = rearrange(k, 'b (head num c) -> head b num c', c=self.head_dim, num=self.k_num), rearrange(v, 'b (head num c) -> head b num c', c=self.head_dim, num=self.k_num)
       sim = torch.einsum('hbpc,hbqc->hbpq', q, k) * self.scale
       probs = nn.functional.softmax(sim, dim=-1)
       output = torch.einsum('hbij,hbjc->hbic', probs, v)
       output = rearrange(output, 'h b p c -> b p (h c)')
       output = self.embedding_muti_head(output)
       output = rearrange(output, 'b (h w) c -> b h w c', h=h_size)
       return output

class Mix(nn.Module):
    def __init__(self, input_dim, text_dim=512):
        super(Mix, self).__init__()
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.input_dim),
            nn.GELU(),
            nn.Linear(self.input_dim, self.input_dim // 2),
        )
        self.cnn1 = conv3x3(self.input_dim, self.input_dim//2, 1)
        self.cnn2 = conv3x3(self.input_dim // 2, self.input_dim // 2, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 4, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2, bias=True),
            nn.Sigmoid()
        )
        self.sg = nn.Sigmoid()


    def forward(self, x, y):
        """
       Args:
           x: input tensor with shape of [b h w c], y: [b text]
       """
        y = self.text_embedding(y)
        x = Rearrange('b h w c -> b c h w')(x)
        y = y * self.sg(self.fc(self.gap(self.cnn2(self.cnn1(x) * (y.unsqueeze(dim=2).unsqueeze(dim=3)))).squeeze(dim=2).squeeze(dim=2)))
        return y


class CATB(nn.Module):
    """ cross attention based transformer model
    """
    def __init__(self, input_dim, output_dim, head_dim, drop_path, text_dim, use_mix=True):
        super(CATB, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.drop_path = drop_path
        self.use_mix = use_mix

        if use_mix:
            self.Mix = Mix(input_dim, text_dim)
        else:
            self.Mix = nn.Sequential(
                nn.Linear(text_dim, self.input_dim),
                nn.GELU(),
                nn.Linear(self.input_dim, self.input_dim // 2),
            )
        self.LN1 = nn.LayerNorm(input_dim)
        self.CA = CA(input_dim, input_dim, head_dim, text_dim=text_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.LN2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x, y):
        if self.use_mix:
            y = self.Mix(x, y)
        else:
            y = self.Mix(y)
        x = x + self.drop_path(self.CA(self.LN1(x), y))
        x = x + self.drop_path(self.mlp(self.LN2(x)))
        return x
    

class Tran_Block(nn.Module):
    def __init__(self, channels, head_dim, drop, text_dim, use_mix):
        """ CA Transformer model
        """
        super(Tran_Block, self).__init__()
        self.channels = channels
        self.head_dim = head_dim
        self.drop = drop
        
        self.cnn1 = conv3x3(self.channels, self.channels) # ResidualBlock(self.channels, self.channels)
        self.CA = CATB(self.channels, self.channels, self.head_dim, self.drop, text_dim, use_mix)
        self.cnn2 = ResidualBlock(self.channels, self.channels)

    def forward(self, x, y):
        x_res = self.cnn1(x)
        x_res = Rearrange('b c h w -> b h w c')(x_res)
        x_res = self.CA(x_res, y)
        x_res = Rearrange('b h w c -> b c h w')(x_res)
        x_res = self.cnn2(x_res)
        return x + x_res, y

class CAtten(AttentionBlock):
    """ the attention model in entropy model
    """
    def __init__(self, input_dim, output_dim, head_dim, drop, text_dim, inter_dim=192, use_mix=True):
        super().__init__(N=inter_dim)
        self.in_conv = conv1x1(input_dim, inter_dim)
        self.out_conv = conv1x1(inter_dim, output_dim)
        self.non_local_block = CATB(inter_dim, inter_dim, head_dim, drop, text_dim, use_mix)

    def forward(self, x, text):
        x = self.in_conv(x)
        z = Rearrange('b h w c -> b c h w')(self.non_local_block(Rearrange('b c h w -> b h w c')(x), text))
        a = self.conv_a(x)    # RB*3
        b = self.conv_b(z)    # RB*3 + conv1x1
        out = a * torch.sigmoid(b)
        out = self.out_conv(out + x)
        return out

class G_A(nn.Module):
    """  the function of g_a
    """
    def __init__(self, config, channels, head_dim, drop, M, text_dim, use_mix) -> None:
        super().__init__()
        self.m0 = ResidualBlockWithStride(3, channels, 2)
        self.m1 = mySequential(*[Tran_Block(channels, head_dim[0], drop, text_dim, use_mix) for i in range(config[0])])
        self.down1 = ResidualBlockWithStride(channels, channels, stride=2)
        self.m2 = mySequential(*[Tran_Block(channels, head_dim[1], drop, text_dim, use_mix) for i in range (config[1])] )
        self.down2 = ResidualBlockWithStride(channels, channels, stride=2)
        self.m3 = mySequential(*[Tran_Block(channels, head_dim[2], drop, text_dim, use_mix) for i in range (config[2])])
        self.d3 = conv3x3(channels, M, 2)
        

    def forward(self, x, text):
        x, _ = self.m1(self.m0(x), text)
        x = self.down1(x)
        x, _ = self.m2(x, text)
        x = self.down2(x)
        x, _ = self.m3(x, text)
        x = self.d3(x)

        return x
    
class G_S(nn.Module):
    """  the funtion of g_s
    """
    def __init__(self, config, channels, head_dim, drop, M, text_dim, use_mix) -> None:
        super().__init__()
        self.m0 = ResidualBlockUpsample(M, channels, 2)
        self.m1 = mySequential(*[Tran_Block(channels, head_dim[0], drop, text_dim, use_mix) for i in range(config[0])])
        self.up1 = ResidualBlockUpsample(channels, channels, 2)
        self.m2 = mySequential(*[Tran_Block(channels, head_dim[1], drop, text_dim, use_mix) for i in range(config[1])])
        self.up2 = ResidualBlockUpsample(channels, channels, 2)
        self.m3 = mySequential(*[Tran_Block(channels, head_dim[2], drop, text_dim, use_mix) for i in range(config[2])])
        self.u3 = subpel_conv3x3(channels, 3, 2)
        

    def forward(self, x, text):
        x, _ = self.m1(self.m0(x), text)
        x = self.up1(x)
        x, _ = self.m2(x, text)
        x = self.up2(x)
        x, _ = self.m3(x, text)
        x = self.u3(x)
        
        return x   

class CM_GRU(CompressionModel):
    def __init__(self, config=[1, 1, 1, 1, 1, 1], head_dim=[8, 16, 32, 32, 16, 8], channels=256, M1=320, M2=192, num_slices=5, use_mix=True, max_support_slices=4, drop_path_rate=0, text_dim=512, **kwargs):
        super().__init__(entropy_bottleneck_channels=channels)
        self.channels = channels
        self.config = config
        self.text_dim = text_dim
        self.head_dim = head_dim
        self.M1, self.M2 = M1, M2
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        
        self.g_a = G_A(self.config[:3], channels, self.head_dim[:3], drop_path_rate, self.M1, self.text_dim, use_mix)
        self.g_s = G_S(self.config[3:], channels, self.head_dim[3:], drop_path_rate, self.M1, self.text_dim, use_mix)
        
        # ------------------- h_a ---------------------- #
        self.ha_d1 = ResidualBlockWithStride(self.M1, self.channels, 2)
        self.ha_CAT = Tran_Block(self.channels, 32, 0., self.text_dim, use_mix)
        self.ha_d2 = conv3x3(self.channels, self.M2, 2)
        
        # ------------------ mean、scale(h_s) ----------------- #
        self.hs_mean_u1 = ResidualBlockUpsample(self.M2, self.channels, 2)
        self.hs_mean_CAT = Tran_Block(self.channels, 32, 0., self.text_dim, use_mix)
        self.hs_mean_u2 = subpel_conv3x3(self.channels, self.M1, 2)
        self.hs_scale_u1 = ResidualBlockUpsample(self.M2, self.channels, 2)
        self.hs_scale_CAT = Tran_Block(self.channels, 32, 0., self.text_dim, use_mix)
        self.hs_scale_u2 = subpel_conv3x3(self.channels, self.M1, 2)
       
        # ------------------ GRU attention ----------------- #
        self.mean_gru = ConvGRU(self.M1, self.M1//self.num_slices)
        self.scale_gru = ConvGRU(self.M1, self.M1//self.num_slices)

        self.atten_mean = CAtten(self.M1, self.M1, 16, 0, self.text_dim, inter_dim=self.M2)
        self.atten_scale = CAtten(self.M1, self.M1, 16, 0, self.text_dim, inter_dim=self.M2)

        # ------------------ Net --------------------- #
        self.final_mean = nn.Sequential(
            conv(self.M1, 224, stride=1, kernel_size=3),
            nn.GELU(),
            conv(224, 128, stride=1, kernel_size=3),
            nn.GELU(),
            conv(128, self.M1 // self.num_slices, stride=1, kernel_size=3),
        )
        self.final_scale = nn.Sequential(
            conv(self.M1, 224, stride=1, kernel_size=3),
            nn.GELU(),
            conv(224, 128, stride=1, kernel_size=3),
            nn.GELU(),
            conv(128, self.M1 // self.num_slices, stride=1, kernel_size=3),
        )
        # ------------------ LRP --------------------- #
        self.lrp = nn.Sequential(
            conv(self.M1 * 2 + (self.M1 // self.num_slices), 512, stride=1, kernel_size=3),
            nn.GELU(),
            conv(512, 352, stride=1, kernel_size=3),
            nn.GELU(),
            conv(352, 192, stride=1, kernel_size=3),
            nn.GELU(),
            conv(192, (self.M1 // self.num_slices), stride=1, kernel_size=3),
        )

        self.entropy_bottleneck = EntropyBottleneck(self.M2)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, text):
        # text = torch.Tensor(x.shape[0],512).to('cuda')
        y = self.g_a(x, text)

        z, _ = self.ha_CAT(self.ha_d1(y), text)
        z = self.ha_d2(z)
        _, z_likelihoods = self.entropy_bottleneck(z)
        
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_means, _ = self.hs_mean_CAT(self.hs_mean_u1(z_hat), text)
        latent_means = self.hs_mean_u2(latent_means)
        latent_scales, _ = self.hs_scale_CAT(self.hs_scale_u1(z_hat), text)
        latent_scales = self.hs_scale_u2(latent_scales)
        
        y = y.chunk(self.num_slices, 1)
        y_hat_slices, y_likehood = [], []
        mean_list, scale_list = [], []
        mean_hidden, scale_hidden = latent_means, latent_scales
        for idx, y_slice in enumerate(y):
            if idx:
                mean_hidden = self.mean_gru(mean_hidden, y_hat_slice)
                scale_hidden = self.scale_gru(scale_hidden, y_hat_slice)
            mean_support = self.atten_mean(mean_hidden, text)
            mean = self.final_mean(mean_support)
            mean_list.append(mean)
            scale_support = self.atten_scale(scale_hidden, text)
            scale = self.final_scale(scale_support)
            scale_list.append(scale)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mean)
            y_likehood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mean) + mean
            lrp_support = torch.cat([mean_hidden, scale_hidden, y_hat_slice], dim=1)
            lrp = torch.tanh(self.lrp(lrp_support)) * 0.5
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mean_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likehood, dim=1)
        x_hat = self.g_s(y_hat, text)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }
    
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net
    
    def compress(self, x, text):
        y = self.g_a(x, text)

        z, _ = self.ha_CAT(self.ha_d1(y), text)
        z = self.ha_d2(z)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_means, _ = self.hs_mean_CAT(self.hs_mean_u1(z_hat), text)
        latent_means = self.hs_mean_u2(latent_means)
        latent_scales, _ = self.hs_scale_CAT(self.hs_scale_u1(z_hat), text)
        latent_scales = self.hs_scale_u2(latent_scales) 

        y = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        mean_list, scale_list = [], []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        mean_hidden, scale_hidden = latent_means, latent_scales

        for idx, y_slice in enumerate(y):
            if idx:
                mean_hidden = self.mean_gru(mean_hidden, y_hat_slice)
                scale_hidden = self.scale_gru(scale_hidden, y_hat_slice)
            mean_support = self.atten_mean(mean_hidden, text)
            mean = self.final_mean(mean_support)
            scale_support = self.atten_scale(scale_hidden, text)
            scale = self.final_scale(scale_support)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mean)
            y_hat_slice = y_q_slice + mean

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_hidden, scale_hidden, y_hat_slice], dim=1)
            lrp = torch.tanh(self.lrp(lrp_support)) * 0.5
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            scale_list.append(scale)
            mean_list.append(mean)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
        
    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood
    
    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)
    
    def decompress(self, strings, text, shape):

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        latent_means, _ = self.hs_mean_CAT(self.hs_mean_u1(z_hat), text)
        latent_means = self.hs_mean_u2(latent_means)
        latent_scales, _ = self.hs_scale_CAT(self.hs_scale_u1(z_hat), text)
        latent_scales = self.hs_scale_u2(latent_scales) 

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        mean_hidden, scale_hidden = latent_means, latent_scales

        for slice_index in range(self.num_slices):
            if slice_index:
                mean_hidden = self.mean_gru(mean_hidden, y_hat_slice)
                scale_hidden = self.scale_gru(scale_hidden, y_hat_slice)
            mean_support = self.atten_mean(mean_hidden, text)
            mean = self.final_mean(mean_support)
            scale_support = self.atten_scale(scale_hidden, text)
            scale = self.final_scale(scale_support)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mean)

            lrp_support = torch.cat([mean_hidden, scale_hidden, y_hat_slice], dim=1)
            lrp = self.lrp(lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat, text).clamp_(0, 1)

        return {"x_hat": x_hat}

    