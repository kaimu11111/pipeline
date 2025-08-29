# 1. Imports ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

F = torch.nn.functional

# 2. CUDA source (kernel + host wrapper) ─────────────────────────────
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel_float(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ out,
                                 int64_t numel){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numel){
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b){
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(),     "tensor sizes must match");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "only float32 supported");

    auto out      = torch::empty_like(a);
    const int64_t n = a.numel();
    const int     t = 256;
    const int     bsz = (n + t - 1) / t;
    add_kernel_float<<<bsz, t>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
    return out;
}
"""

# 3. C++ prototypes  ─────────────────────────────────────────────────
cpp_src = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"

# 4. build / load extension ─────────────────────────────────────────
_add_mod = load_inline(
    name       = "custom_add",
    cpp_sources= cpp_src,
    cuda_sources = source,
    functions  = ["add_cuda"],
    verbose    = False,
)

# 5. Model Definition ───────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self, img_size=112, patch_size=4, in_chans=3,
                 num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False,
                 pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers   = len(depths)
        self.num_classes  = num_classes
        self.embed_dim    = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim,
            norm_layer if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution

        self.pos_drop = nn.Dropout(drop_rate)

        # stochastic-depth schedule
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()

        self.layers = nn.ModuleList()
        dp_offset = 0
        for i_layer in range(self.num_layers):
            depth_i = depths[i_layer]
            layer = BasicLayer(
                dim       = int(embed_dim * 2 ** i_layer),
                input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                    patches_resolution[1] // (2 ** i_layer)),
                depth     = depth_i,
                num_heads = num_heads[i_layer],
                window_size = window_size,
                mlp_ratio   = mlp_ratio,
                qkv_bias    = qkv_bias,
                drop        = drop_rate,
                attn_drop   = attn_drop_rate,
                drop_path   = dpr[dp_offset: dp_offset + depth_i],
                norm_layer  = norm_layer,
                downsample  = PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint = use_checkpoint,
                pretrained_window_size = pretrained_window_sizes[i_layer],
            )
            dp_offset += depth_i
            self.layers.append(layer)

        self.norm    = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    # ------------------------------------------------------------------
    def forward_features(self, x):
        x = self.patch_embed(x)          # B, L, C
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                 # B, L, C
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))

# ────────────────────────────────────────────────────────────────────
# Helper utilities & sub-modules (appear *after* ModelNew so import-order
# rule is respected)                                                  
# ────────────────────────────────────────────────────────────────────
def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob   = 1.0 - self.drop_prob
        shape       = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
                .view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1) \
               .permute(0, 1, 3, 2, 4, 5).contiguous() \
               .view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=(0, 0)):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size       # (Wh, Ww)
        self.num_heads   = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), True)

        # relative position encoding (continuous)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, True), nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, False)
        )

        relative_coords_h = torch.arange(-(window_size[0] - 1),
                                          window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(window_size[1] - 1),
                                          window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid(relative_coords_h, relative_coords_w, indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1,2*Ww-1,2

        if pretrained_window_size[0] > 0:
            relative_coords_table[..., 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[..., 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[..., 0] /= (window_size[0] - 1)
            relative_coords_table[..., 1] /= (window_size[1] - 1)

        relative_coords_table *= 8  # scale
        relative_coords_table = torch.sign(relative_coords_table) * \
            torch.log2(torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8.0))
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2,Wh,Ww
        coords_flatten = torch.flatten(coords, 1)                                   # 2,Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[..., 0] += window_size[0] - 1
        relative_coords[..., 1] += window_size[1] - 1
        relative_coords[..., 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                           # Wh*Ww,Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(100.))).exp()
        attn = attn * logit_scale

        bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = bias_table[self.relative_position_index.view(-1)] \
                                    .view(self.window_size[0]*self.window_size[1],
                                          self.window_size[0]*self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + 16 * torch.sigmoid(relative_position_bias).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size if min(input_resolution) > window_size else 0

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, to_2tuple(window_size), num_heads,
            qkv_bias, attn_drop, drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W      = input_resolution
            img_mask  = torch.zeros(1, H, W, 1)  # 1 H W 1
            h_slices  = (slice(0, -window_size),
                         slice(-window_size, -shift_size),
                         slice(-shift_size, None))
            w_slices  = (slice(0, -window_size),
                         slice(-window_size, -shift_size),
                         slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size*window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                   .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        x_windows   = window_partition(shifted_x, self.window_size)      # nW*B, win, win, C
        x_windows   = x_windows.view(-1, self.window_size*self.window_size, C)
        attn_windows = self.attn(x_windows, self.attn_mask)              # nW*B, win^2, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x   = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x
        x = x.view(B, H*W, C)

        x = fused_add(shortcut, self.drop_path(self.norm1(x)))
        x = fused_add(x,        self.drop_path(self.norm2(self.mlp(x))))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm      = norm_layer(4*dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W and H % 2 == 0 and W % 2 == 0, "Resolution mismatch"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4*C)
        x  = self.norm(x)
        x  = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 window_size, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, input_resolution, num_heads, window_size,
                shift_size=0 if (i % 2 == 0) else window_size//2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim, norm_layer) if downsample else None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

# fused element-wise addition (CUDA)  ────────────────────────────────
class _FusedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return _add_mod.add_cuda(a.contiguous(), b.contiguous())
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, grad_out

def fused_add(a, b):
    return _FusedAdd.apply(a, b)

# --------------------------------------------------------------------
# helper functions required by external harness
# --------------------------------------------------------------------
batch_size = 5
image_size = 112

def get_inputs():
    return [torch.rand(batch_size, 3, image_size, image_size, device='cuda')]

def get_init_inputs():
    return []
