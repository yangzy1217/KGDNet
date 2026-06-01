import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .FastSAM.fastsam import FastSAM
from utils import initialize_weights


class DWConv(nn.Module):
    """Depthwise separable convolution with fewer parameters."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.point(self.depth(x)))


class GeoPE(nn.Module):
    """2D rotary-style positional encoding generated for arbitrary spatial sizes."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, (dim + 1) // 2).float() / dim))

    def _compute_angles(self, coords):
        """Compute positional angle features."""
        sinusoid = torch.einsum('i,j->ij', coords.float(), self.inv_freq.to(coords.device))
        sin_part = sinusoid.sin()[:, :(self.dim + 1) // 2]
        cos_part = sinusoid.cos()[:, :self.dim // 2]
        return torch.cat([sin_part, cos_part], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_coords, y_coords = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='xy'
        )
        x_emb = self._compute_angles(x_coords.reshape(-1))
        y_emb = self._compute_angles(y_coords.reshape(-1))
        emb = (x_emb + y_emb).view(H, W, self.dim)
        return x * (1.0 + emb.permute(2, 0, 1).unsqueeze(0))


class MultiHeadLatentSpatioTemporalAttention(nn.Module):
    """Multi-head attention with latent spatial compression and optional RoPE."""

    def __init__(self, embed_dim, num_heads=8, compression_ratio=4, use_rope: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.compressed_dim = embed_dim // compression_ratio
        self.use_rope = use_rope

        self.W_DKV = nn.Conv2d(embed_dim, self.compressed_dim, 1)
        self.W_UK = nn.Linear(self.compressed_dim, num_heads * self.head_dim)
        self.W_UV = nn.Linear(self.compressed_dim, num_heads * self.head_dim)

        self.W_KR = nn.Conv2d(embed_dim, num_heads * self.head_dim, 1)
        self.W_QR = nn.Conv2d(embed_dim, num_heads * self.head_dim, 1)

        self.rotary_pe = GeoPE(self.head_dim) if use_rope else None
        self.W_O = nn.Conv2d(num_heads * self.head_dim, embed_dim, 1)

        nn.init.xavier_uniform_(self.W_UK.weight)
        nn.init.xavier_uniform_(self.W_UV.weight)
        nn.init.zeros_(self.W_UK.bias)
        nn.init.zeros_(self.W_UV.bias)

    def forward(self, query, key_value=None):
        key_value = key_value if key_value is not None else query
        B, C, H, W = query.shape

        ckv = self.W_DKV(key_value)
        flat_ckv = ckv.permute(0, 2, 3, 1).reshape(-1, self.compressed_dim)

        K = self.W_UK(flat_ckv).view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_UV(flat_ckv).view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        Q = self.W_QR(query)

        if self.use_rope and (self.rotary_pe is not None):
            QR = self._apply_rotary_pe(Q, B, H, W)
            Q = Q.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            KR = self._apply_rotary_pe_heads(K)
            QR = torch.cat([QR, Q], dim=-1)
            KR = torch.cat([KR, Q], dim=-1)
        else:
            QR = Q.view(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            KR = K

        attn = torch.einsum('bnhwd,bnhwk->bnhwk', QR, KR) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bnhwk,bnhwd->bnhwd', attn, V)
        out = out.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        out = self.W_O(out)

        return out + query

    def _apply_rotary_pe(self, x, B, H, W):
        """Apply 2D positional encoding to a packed multi-head tensor."""
        assert self.rotary_pe is not None, "rotary_pe is None but _apply_rotary_pe was called"
        x = x.view(B, self.num_heads, self.head_dim, H, W)
        x = x.permute(0, 1, 3, 4, 2).reshape(-1, self.head_dim, H, W)
        return self.rotary_pe(x).view(B, self.num_heads, H, W, self.head_dim)

    def _apply_rotary_pe_heads(self, x):
        """Apply 2D positional encoding to a [B, heads, H, W, head_dim] tensor."""
        assert self.rotary_pe is not None, "rotary_pe is None but _apply_rotary_pe_heads was called"
        B, n, H, W, d = x.shape
        x4 = x.permute(0, 1, 4, 2, 3).reshape(B * n, d, H, W)
        x4 = self.rotary_pe(x4)
        return x4.view(B, n, d, H, W).permute(0, 1, 3, 4, 2)


class RepDWDown(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw = nn.Conv2d(C, C, 3, stride=2, padding=1, groups=C, bias=False)
        self.dw_bn = nn.BatchNorm2d(C)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(C)
        self.skip = nn.Conv2d(C, C, 1, stride=2, bias=False)
        self.skip_bn = nn.BatchNorm2d(C)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.pw_bn(self.pw(self.dw_bn(self.dw(x))))
        y = y + self.skip_bn(self.skip(x))
        return self.act(y)


class GAFM(nn.Module):
    """Guided Alignment Fusion Module."""

    def __init__(self, in_dim, num_heads=8, compression_ratio=4):
        super().__init__()
        self.attn = MultiHeadLatentSpatioTemporalAttention(in_dim, num_heads, compression_ratio, use_rope=True)
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.GELU()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, max(in_dim // 4, 16), 1), nn.GELU(),
            nn.Conv2d(max(in_dim // 4, 16), in_dim, 1), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x1, x2):
        orig_x1 = x1
        x1_down = self.down(x1)
        x2_down = self.down(x2)

        attn_out = self.attn(x2_down, x1_down)
        up_out = self.up(attn_out)

        g = self.gate(up_out)
        out = orig_x1 + g * up_out
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out


class DecoderFusionBlock(nn.Module):
    """Decoder block with upsampling, fusion, and optional paired-feature adaptation."""

    def __init__(self, in_high, in_low, out_ch, paired_feat_channels=None):
        super().__init__()
        self.fusion = nn.Sequential(
            DWConv(in_high + in_low, out_ch, 3, padding=1),
            nn.GELU(),
            DWConv(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.paired_proj = self._create_projection(paired_feat_channels, out_ch)
        self.adaptor = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def _create_projection(self, in_ch, out_ch):
        """Create feature projection layer."""
        if in_ch and in_ch != out_ch:
            return nn.Conv2d(in_ch, out_ch, kernel_size=1)
        return nn.Identity()

    def forward(self, x, low, paired_feat=None):
        up_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        concat = torch.cat([up_x, low], dim=1)
        fused = self.fusion(concat)

        if paired_feat is not None:
            proj_feat = self.paired_proj(paired_feat)
            fused = fused + self.adaptor(proj_feat)

        return fused


class RRB(nn.Module):
    """Residual Refine Block."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.downsample = self._create_downsample(inplanes, planes, stride)

    def _create_downsample(self, inplanes, planes, stride):
        """Create the residual downsampling path when needed."""
        if stride != 1 or inplanes != planes:
            return nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride),
                nn.BatchNorm2d(planes)
            )
        return None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ClassificationHeadMain(nn.Module):
    def __init__(self, in_ch=64, mid_ch=32, out_ch=1, use_sigmoid_at_infer=False):
        super().__init__()
        self.use_sigmoid_at_infer = use_sigmoid_at_infer
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, max(mid_ch // 2, 16), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(mid_ch // 2, 16)),
            nn.GELU(),
            nn.Conv2d(max(mid_ch // 2, 16), out_ch, kernel_size=1)
        )

    def forward(self, x):
        logits = self.head(x)
        if (not self.training) and self.use_sigmoid_at_infer:
            return torch.sigmoid(logits)
        return logits


class KGDNet(nn.Module):
    """Main change-detection network."""

    def __init__(
            self,
            num_embed=8,
            model_name='FastSAM-x.pt',
            model_depth_name='vits',
            device='cuda',
            conf=0.4,
            iou=0.9,
            imgsz=1024,
            retina_masks=True
    ):
        super().__init__()
        self.fast_sam = FastSAM(model_name)
        self.device = device
        self.model_depth_name = model_depth_name
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.save_vis = False
        self.vis_cache = {}

        self.GAFM_knowledge = nn.ModuleDict({
            's4': GAFM(160),
            's8': GAFM(320),
            's16': GAFM(640),
            's32': GAFM(640)
        })

        self.FRM = nn.Sequential(
            RRB(160, 160),
            RRB(160, 80),
            RRB(80, 64),
        )
        self.decoder = self.UpsampleDecoder()

        self.GAFM_temporal = nn.ModuleDict({
            's4': GAFM(160),
            's8': GAFM(320),
            's16': GAFM(640),
            's32': GAFM(640)
        })
        self.DPM = nn.Sequential(
            nn.Conv2d(80, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head = ClassificationHeadMain(64)

        self._freeze_backbone()
        initialize_weights(self.GAFM_knowledge, self.FRM, self.decoder, self.GAFM_temporal, self.head, self.DPM)

    def UpsampleDecoder(self):
        """Create decoder branches for both temporal streams and the difference stream."""
        decoder = nn.ModuleDict()
        for phase in ['t1', 't2', 'diff']:
            decoder[phase] = nn.ModuleDict({
                'dec2': DecoderFusionBlock(640, 640, 320, paired_feat_channels=640),
                'dec1': DecoderFusionBlock(320, 320, 160, paired_feat_channels=320),
                'dec0': DecoderFusionBlock(160, 160, 80, paired_feat_channels=160)
            })
        return decoder

    def _freeze_backbone(self):
        """Freeze FastSAM backbone parameters."""
        for param in self.fast_sam.model.parameters():
            param.requires_grad = False

    def SAM_encoder(self, x):
        """Extract multi-scale features from FastSAM."""
        feats = self.fast_sam(
            x,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
        )
        return {'s4': feats[3], 's8': feats[0], 's16': feats[1], 's32': feats[2]}

    def GAFM(self, features, depth_map, mode):
        """Fuse knowledge features or align temporal features by scale."""
        processed = {}
        for scale, fea in features.items():
            fea_clone = fea.clone()
            depth_map_clone = depth_map[scale].clone()
            if mode == 'K':
                processed[scale] = self.GAFM_knowledge[scale](fea_clone, depth_map_clone)
            elif mode == 'T':
                processed[scale] = self.GAFM_temporal[scale](fea_clone, depth_map_clone)

        return processed

    def MSDecoder(self, features, decoder, paired_features=None, use_diff=False):
        """Decode multi-scale features."""
        if use_diff:
            return self.diff_decoder(features, decoder)

        d2 = decoder['dec2'](features['s32'], features['s16'], paired_features['s16'])
        d1 = decoder['dec1'](d2, features['s8'], paired_features['s8'])
        d0 = decoder['dec0'](d1, features['s4'], paired_features['s4'])

        return d0

    def diff_decoder(self, features, decoder):
        """Decode absolute-difference features."""
        d2 = decoder['dec2'](features['s32'], features['s16'])
        d1 = decoder['dec1'](d2, features['s8'])
        return decoder['dec0'](d1, features['s4'])

    def forward(self, x1, x2, x1_depth, x2_depth):
        f1_raw = self.SAM_encoder(x1)
        f2_raw = self.SAM_encoder(x2)

        depth1 = self.SAM_encoder(x1_depth)
        depth2 = self.SAM_encoder(x2_depth)

        f1 = self.GAFM(f1_raw, depth1, mode='K')
        f2 = self.GAFM(f2_raw, depth2, mode='K')

        f1_fusion = self.GAFM(f1, f2, mode='T')
        f2_fusion = self.GAFM(f2, f1, mode='T')

        d1 = self.MSDecoder(f1_fusion, self.decoder['t1'], f2_fusion)
        d2 = self.MSDecoder(f2_fusion, self.decoder['t2'], f1_fusion)

        diff_feats = {k: torch.abs(f1[k] - f2[k]) for k in f1.keys()}
        d3 = self.MSDecoder(diff_feats, self.decoder['diff'], use_diff=True)

        gate = self.DPM(d3)
        d1_mod = d1 * gate
        d2_mod = d2 * gate

        low_fea = torch.cat([d1_mod, d2_mod], dim=1)
        refined_fea = self.FRM(low_fea)

        if (not self.training) and getattr(self, "save_vis", False):
            self.vis_cache["d1_mod"] = d1_mod.detach().to("cpu")
            self.vis_cache["d2_mod"] = d2_mod.detach().to("cpu")
            self.vis_cache["refined_fea"] = refined_fea.detach().to("cpu")
            self.vis_cache["gate"] = gate.detach().to("cpu")

        change_map = self.head(refined_fea)
        change_map = F.interpolate(change_map, x1.shape[-2:], mode='bilinear')
        return change_map
