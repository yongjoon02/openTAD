# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmaction.utils import ConfigType, OptConfigType
from mmaction.models.backbones.vit_mae import get_sinusoid_encoding


class Adapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # temporal depth-wise convolution
        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=hidden_dims,
        )
        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape  # 48, 8*10*10, 384
        # h와 w를 확실히 정수로 변환
        if hasattr(h, 'item'):
            h_int = int(h.item())
        elif isinstance(h, (list, tuple)):
            h_int = int(h[1]) if len(h) > 1 else int(h[0])  # 두 번째 값 사용 (실제 공간 차원)
        else:
            h_int = int(h)
            
        if hasattr(w, 'item'):
            w_int = int(w.item())
        elif isinstance(w, (list, tuple)):
            w_int = int(w[1]) if len(w) > 1 else int(w[0])  # 두 번째 값 사용 (실제 공간 차원)
        else:
            w_int = int(w)
            
        attn = x.reshape(-1, self.temporal_size, h_int, w_int, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = attn.unflatten(0, (-1, h_int, w_int)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
        attn = attn.reshape(B, N, C)
        x = x + attn

        x = self.up_proj(x)
        return x * self.gamma + inputs


class PlainAdapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape  # 48, 8*10*10, 384
        # h와 w를 확실히 정수로 변환
        if hasattr(h, 'item'):
            h_int = int(h.item())
        elif isinstance(h, (list, tuple)):
            h_int = int(h[1]) if len(h) > 1 else int(h[0])  # 두 번째 값 사용 (실제 공간 차원)
        else:
            h_int = int(h)
            
        if hasattr(w, 'item'):
            w_int = int(w.item())
        elif isinstance(w, (list, tuple)):
            w_int = int(w[1]) if len(w) > 1 else int(w[0])  # 두 번째 값 사용 (실제 공간 차원)
        else:
            w_int = int(w)
            
        attn = x.reshape(-1, self.temporal_size, h_int, w_int, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = attn.unflatten(0, (-1, h_int, w_int)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
        attn = attn.reshape(B, N, C)
        x = x + attn

        x = self.up_proj(x)
        return x * self.gamma + inputs


class Attention(BaseModule):
    """Multi-head Self-attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, C = x.shape

        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # standard self-attention
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # fast attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """The basic block in the Vision Transformer.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        act_cfg (dict or ConfigDict): Config for activation layer in FFN.
            Defaults to `dict(type='GELU')`.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.use_adapter = use_adapter

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if self.use_adapter:
            self.adapter = Adapter(
                embed_dims=embed_dims,
                kernel_size=3,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
            )

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the transformer block, same size as inputs.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            if self.use_adapter:
                x = self.adapter(x, h, w)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformerAdapter(BaseModule):
    """Vision Transformer with Adapter for Video Action Detection.

    Args:
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Patch size. Defaults to 16.
        in_channels (int): Number of input channels. Defaults to 3.
        embed_dims (int): Embedding dimension. Defaults to 768.
        depth (int): Depth of transformer. Defaults to 12.
        num_heads (int): Number of attention heads. Defaults to 12.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Defaults to True.
        qk_scale (int): Override default qk scale of head_dim ** -0.5 if set.
            Defaults to None.
        drop_rate (float): Dropout rate. Defaults to 0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='LN', eps=1e-6).
        num_frames (int): Number of frames per attention. Defaults to 16.
        tubelet_size (int): Size of tubelet. Defaults to 2.
        use_mean_pooling (bool): Whether to use mean pooling. Defaults to True.
        pretrained (str): Path to pretrained model. Defaults to None.
        return_feat_map (bool): Whether to return feature map. Defaults to False.
        with_cp (bool): Use checkpoint or not. Defaults to False.
        adapter_mlp_ratio (float): MLP ratio for adapter. Defaults to 0.25.
        total_frames (int): Total number of frames. Defaults to 768.
        adapter_index (list): Index of layers to add adapter. Defaults to [3, 5, 7, 11].
        init_cfg (dict): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,  # frames per attention
        tubelet_size: int = 2,
        use_mean_pooling: int = True,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        with_cp: bool = False,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_cfg = norm_cfg
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_mean_pooling = use_mean_pooling
        self.pretrained = pretrained
        self.return_feat_map = return_feat_map
        self.with_cp = with_cp
        self.adapter_mlp_ratio = adapter_mlp_ratio
        self.total_frames = total_frames
        self.adapter_index = adapter_index
        self.use_abs_pos_emb = True # Added missing attribute

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),  # 3D conv이므로 3개 값 필요
            norm_cfg=None,
        )

        # positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, (total_frames // tubelet_size) * (img_size // patch_size) ** 2, embed_dims))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    use_adapter=(i in adapter_index),
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    temporal_size=total_frames // tubelet_size,
                )
            )

        # final norm
        self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # load pretrained weights
        if pretrained:
            self._freeze_layers()

        # Initialize mean and std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1))
        
        # Initialize weights
        self.apply(self._init_weights)
        trunc_normal_init(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                constant_init(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            constant_init(m.bias, 0)
            constant_init(m.weight, 1.0)

    def forward(self, x):
        # Input: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Basic normalization
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        
        # Patch embedding expects [B, C, T, H, W] format
        patch_result = self.patch_embed(x)
        if isinstance(patch_result, tuple):
            x, patch_info = patch_result
        else:
            x = patch_result
            patch_info = None
        
        # Add position embedding
        if hasattr(self, 'pos_embed') and self.pos_embed is not None:
            if hasattr(self, 'use_abs_pos_emb') and self.use_abs_pos_emb:
                # Resize pos_embed to match x size
                if x.shape[1] != self.pos_embed.shape[1]:
                    pos_embed = self.pos_embed
                    if hasattr(self, 'cls_token') and self.cls_token is not None:
                        pos_embed = pos_embed[:, 1:]  # Remove cls_token position
                    pos_embed = pos_embed[:, :x.shape[1]]
                    x = x + pos_embed
                else:
                    x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            # Calculate spatial dimensions for adapter
            B, N, C = x.shape
            T_patches = T // self.tubelet_size  # temporal patches
            H_patches = W_patches = int((N // T_patches) ** 0.5)  # spatial patches (assuming square)
            
            x = block(x, H_patches, W_patches)
        
        # Apply final normalization
        if hasattr(self, 'fc_norm'):
            x = self.fc_norm(x)
        
        # Return format based on return_feat_map
        if self.return_feat_map:
            # (1) cls 토큰 제거
            if getattr(self, 'cls_token', None) is not None:
                x = x[:, 1:]           # [B, N-1, C]

            # (2) 시퀀스를 [B, C, T] 형태로 변환
            B, N, C = x.shape
            T_p = T // self.tubelet_size              # 예: 768/2 = 384
            S   = N // T_p                            # spatial 패치 수

            x = x.reshape(B, T_p, S, C)               # [B, T_p, S, C]
            x = x.mean(dim=2)                         # [B, T_p, C] - spatial 차원 평균
            x = x.permute(0, 2, 1)                    # [B, C, T_p]

            # BackboneWrapper가 기대하는 형태: [B*N, C, T]
            # 입력이 [B*N, C, T, H, W]이므로 출력도 [B*N, C, T]여야 함
            return x

        elif self.use_mean_pooling:
            # mean pooling
            if hasattr(self, 'cls_token') and self.cls_token is not None:
                x = x[:, 1:].mean(dim=1)  # Remove cls_token and mean pool
            else:
                x = x.mean(dim=1)
            return x
        else:
            return x

    def _freeze_layers(self):
        """Freeze layers for pretrained model."""
        for name, param in self.named_parameters():
            if "adapter" not in name:  # adapter는 학습 가능하게 유지
                param.requires_grad = False
