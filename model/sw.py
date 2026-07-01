import torch
from torch import nn
import torchvision
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=k_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                          # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))# [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)         # [B, C, 1, 1]
        return x * self.sigmoid(y)


class FeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()

        # Branch 1: 深度可分离卷积（替换原 15x15 大核分组卷积）
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=3, stride=2,
                padding=1, groups=512, bias=False),   # depthwise
            nn.Conv2d(512, 512, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Branch 2: 串联空洞卷积（替换原 11x11 单次大核卷积）
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                1024, 512, kernel_size=3,
                padding=2, dilation=2, bias=False),
            nn.Conv2d(
                512, 512, kernel_size=3,
                padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Branch 3: 双线性插值上采样（替换原最近邻上采样）
        self.branch3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # 空间注意力：CBAM 风格（均值 + 最大值，替换原单均值）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 通道注意力：ECA（替换原 SE Block）
        self.eca = ECA(channels=512)

        # Branch3 与 Branch2 残差对齐（新增）
        self.align = nn.Conv2d(512, 512, kernel_size=1)

        # 融合输出（替换原 Concat+Conv）
        self.fusion_norm = nn.BatchNorm2d(512)
        self.out_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.GELU(),
        )

    def forward(self, x1, x2, x3):
        x1_1 = self.branch1(x1)   # [B, 512, H, W]
        x2_1 = self.branch2(x2)   # [B, 512, H, W]

        # Branch3 上采样后注入 Branch2 中层特征（残差语义对齐）
        x3_1 = self.branch3(x3) + self.align(x2_1)  # [B, 512, H, W]

        # Detail 分支：CBAM 空间注意力
        avg_s = torch.mean(x1_1, dim=1, keepdim=True)     # [B, 1, H, W]
        max_s, _ = torch.max(x1_1, dim=1, keepdim=True)   # [B, 1, H, W]
        spatial_w = self.spatial_attn(
            torch.cat([avg_s, max_s], dim=1))              # [B, 1, H, W]
        detail = spatial_w * x2_1                          # [B, 512, H, W]

        # Seman 分支：ECA 通道注意力
        seman = self.eca(x3_1) * x2_1                     # [B, 512, H, W]

        # 融合：Hadamard 乘积 + 加法残差（替换原 Concat+Conv）
        inter = detail * seman    # 协同激活区域
        union = detail + seman    # 保留各自独立信息
        out = self.out_conv(self.fusion_norm(inter + union))  # [B, 1024, H, W]

        return out

class Encoder(nn.Module):


    def __init__(self,
                 NetType='swin_small',
                 encoded_image_size=14,
                 attention_method="ByPixel",
                 img_size=256):  # 改为256作为默认值
        super().__init__()

        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method
        self.net_type = NetType
        self.img_size = img_size

        swin_model_name = 'swin_small_patch4_window7_224'

        # 测试实际输出
        temp_model = timm.create_model(
            swin_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=img_size  # 使用传入的img_size
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            test_features = temp_model(dummy_input)
            actual_channels = [f.shape[-1] for f in test_features]

        del temp_model

        self.stage2_channels = actual_channels[0]
        self.stage3_channels = actual_channels[1]
        self.stage4_channels = actual_channels[2]

        # 创建主干网络 - 关键修改：使用img_size参数
        self.swin_backbone = timm.create_model(
            swin_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=img_size  # 使用传入的img_size
        )

        # 通道对齐层
        self.align_stage2 = nn.Sequential(
            nn.Conv2d(self.stage2_channels, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.align_stage3 = nn.Sequential(
            nn.Conv2d(self.stage3_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.align_stage4 = nn.Sequential(
            nn.Conv2d(self.stage4_channels, 2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        # 特征融合模块
        self.FF = FeatureFusion()

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # 微调设置
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation

        Args:
            images: [B, 3, H, W]

        Returns:
            out: [B, encoded_image_size, encoded_image_size, 1024]
        """
        # 提取多阶段特征
        features = self.swin_backbone(images)
        out2, out3, out4 = features

        # Swin输出: [B, H, W, C] -> [B, C, H, W]
        out2 = out2.permute(0, 3, 1, 2).contiguous()
        out3 = out3.permute(0, 3, 1, 2).contiguous()
        out4 = out4.permute(0, 3, 1, 2).contiguous()

        # 通道对齐
        out2 = self.align_stage2(out2)
        out3 = self.align_stage3(out3)
        out4 = self.align_stage4(out4)

        # 特征融合
        out = self.FF(out2, out3, out4)

        # 自适应池化
        out = self.adaptive_pool(out)

        # 转换为decoder所需格式 [B, H, W, C]
        out = out.permute(0, 2, 3, 1)

        return out

    def fine_tune(self, fine_tune=True):
        """
        微调策略: 冻结早期层，微调后期层

        Args:
            fine_tune: 是否允许微调
        """
        # 冻结Patch Embedding
        if hasattr(self.swin_backbone, 'patch_embed'):
            for param in self.swin_backbone.patch_embed.parameters():
                param.requires_grad = False

        # 分层冻结/解冻
        if hasattr(self.swin_backbone, 'layers'):
            for i, layer in enumerate(self.swin_backbone.layers):
                if i < 2:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = fine_tune

        # 通道对齐层始终可训练
        for module in [self.align_stage2, self.align_stage3, self.align_stage4]:
            for param in module.parameters():
                param.requires_grad = True

        # FeatureFusion始终可训练
        for param in self.FF.parameters():
            param.requires_grad = True


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        #attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2))  # (batch_size, pixels, encoder_dim)
        return attention_weighted_encoding, alpha

class CrossAttention(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=8):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossAttention, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, input1, input2):
        # dif_as_kv
        input1 = input1.permute(1, 0, 2)
        input2 = input2.permute(1, 0, 2)
        output_1 = self.cross1(input1, input2)  # (Q,K,V)
        output_1 = output_1.permute(1, 0, 2)
        return output_1
    def cross1(self, input,input2):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, input2, input2)  # (Q,K,V)
        output = input + self.dropout1(attn_output)
        output = self.activation(self.norm1(output))
        return output


class SpatialAzimuthTokenEncoding(nn.Module):
    """
    Lightweight spatial-azimuth encoding for visual tokens.

    It encodes:
        x, y              : normalized spatial coordinates
        dist              : distance to image center
        sin(theta), cos(theta): azimuth-angle cues

    gamma is initialized as 0, so this module is initially an identity mapping.
    """

    def __init__(self, dim):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(5, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, spatial_shape=None):
        """
        Args:
            x: [B, N, D]
            spatial_shape: optional tuple (H, W)

        Returns:
            x with spatial-azimuth encoding: [B, N, D]
        """
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        if spatial_shape is None:
            H = int(math.sqrt(N))
            W = H
            if H * W != N:
                raise ValueError(
                    f"Cannot infer spatial shape from N={N}. "
                    f"Please provide spatial_shape=(H, W)."
                )
        else:
            H, W = spatial_shape
            if H * W != N:
                raise ValueError(
                    f"spatial_shape {spatial_shape} does not match N={N}."
                )

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij"
        )

        dist = torch.sqrt(xx ** 2 + yy ** 2 + 1e-6)
        theta = torch.atan2(yy, xx)

        pos = torch.stack(
            [
                xx,
                yy,
                dist,
                torch.sin(theta),
                torch.cos(theta)
            ],
            dim=-1
        )  # [H, W, 5]

        pos = pos.view(1, N, 5).expand(B, -1, -1)  # [B, N, 5]

        return x + self.gamma * self.proj(pos)

class TVAttention(nn.Module):
    """
    Improved MSVA / TVAttention module.

    Improvements:
    1. Uses text/word features as query and visual tokens as key/value.
    2. Introduces fine-grained and coarse-grained hierarchical visual tokens.
    3. Adds lightweight spatial-azimuth encoding with gamma=0 initialization.
    4. Keeps the original MSVA output and fuses the enhanced branch through a residual gate.

    Inputs:
        TextFeature:
            Global text feature, preferably decoder hidden state h_t.
            Shape: [B, text_dim]

        wordFeature:
            Word-level feature, preferably previous word embedding e(y_{t-1}).
            Shape: [B, embed_dim]

        VisionFeature:
            HDAF visual tokens.
            Shape: [B, N, encoder_dim], e.g. [B, 196, 1024]
            Also supports [B, H, W, encoder_dim] or [B, encoder_dim, H, W].

    Output:
        fused feature: [B, encoder_dim]
    """

    def __init__(
            self,
            encoder_dim,
            embed_dim,
            attention_dim,
            text_dim=1000,
            num_heads=8,
            dropout=0.1
    ):
        super(TVAttention, self).__init__()

        assert encoder_dim % 2 == 0, "encoder_dim must be divisible by 2."
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads."

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.text_dim = text_dim

        half_dim = encoder_dim // 2

        # =========================================================
        # Original MSVA branch: keep the original behavior
        # =========================================================

        self.text_to_vision = nn.MultiheadAttention(
            half_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.word_to_vision = nn.MultiheadAttention(
            half_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.text_proj = nn.Linear(text_dim, half_dim)
        self.word_proj = nn.Linear(embed_dim, half_dim)

        self.orig_norm_v1 = nn.LayerNorm(half_dim)
        self.orig_norm_v2 = nn.LayerNorm(half_dim)
        self.orig_norm_text = nn.LayerNorm(half_dim)
        self.orig_norm_word = nn.LayerNorm(half_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 4, encoder_dim)
        )

        self.norm3 = nn.LayerNorm(encoder_dim)

        # =========================================================
        # Enhanced MSVA branch
        # =========================================================

        # Visual projection
        self.enh_vision_proj = nn.Linear(encoder_dim, attention_dim)

        # Spatial-azimuth encoding
        self.spatial_azimuth = SpatialAzimuthTokenEncoding(attention_dim)

        # Text and word query projections
        self.enh_text_query_proj = nn.Linear(text_dim, attention_dim)
        self.enh_word_query_proj = nn.Linear(embed_dim, attention_dim)

        self.enh_query_norm = nn.LayerNorm(attention_dim)
        self.enh_vision_norm = nn.LayerNorm(attention_dim)

        # Global semantic-guided visual attention
        self.global_text_to_vision = nn.MultiheadAttention(
            attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Word-level semantic-guided visual attention
        self.word_semantic_to_vision = nn.MultiheadAttention(
            attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Project enhanced semantic-visual feature back to encoder_dim
        self.enh_out_proj = nn.Linear(attention_dim * 2, encoder_dim)

        # Visual residual summary
        self.enh_visual_residual = nn.Linear(encoder_dim, encoder_dim)

        # Gate inside enhanced branch
        self.enh_gate_net = nn.Sequential(
            nn.Linear(encoder_dim * 2 + text_dim + embed_dim, encoder_dim),
            nn.Sigmoid()
        )

        self.enh_norm1 = nn.LayerNorm(encoder_dim)
        self.enh_norm2 = nn.LayerNorm(encoder_dim)

        self.enh_ffn = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 4, encoder_dim)
        )

        # =========================================================
        # Residual fusion between original branch and enhanced branch
        # =========================================================

        self.residual_gate = nn.Sequential(
            nn.Linear(encoder_dim * 2 + text_dim + embed_dim, encoder_dim),
            nn.Sigmoid()
        )

        # gamma=0 makes the whole module initially equivalent to the original MSVA branch
        self.gamma = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

    def _to_tokens(self, VisionFeature):
        """
        Convert visual feature to [B, N, C].

        Supports:
            [B, N, C]
            [B, H, W, C]
            [B, C, H, W]
        """
        if VisionFeature.dim() == 3:
            B, N, C = VisionFeature.shape
            if C != self.encoder_dim:
                raise ValueError(
                    f"Expected encoder_dim={self.encoder_dim}, but got C={C}."
                )

            H = int(math.sqrt(N))
            W = H if H * H == N else None
            spatial_shape = (H, W) if W is not None else None

            return VisionFeature, spatial_shape

        elif VisionFeature.dim() == 4:
            # Case 1: [B, H, W, C]
            if VisionFeature.shape[-1] == self.encoder_dim:
                B, H, W, C = VisionFeature.shape
                tokens = VisionFeature.view(B, H * W, C).contiguous()
                return tokens, (H, W)

            # Case 2: [B, C, H, W]
            elif VisionFeature.shape[1] == self.encoder_dim:
                B, C, H, W = VisionFeature.shape
                tokens = VisionFeature.flatten(2).transpose(1, 2).contiguous()
                return tokens, (H, W)

            else:
                raise ValueError(
                    "For 4D VisionFeature, expected shape [B, H, W, C] "
                    "or [B, C, H, W]."
                )

        else:
            raise ValueError(
                "VisionFeature must be a 3D or 4D tensor."
            )

    def _build_hierarchical_tokens(self, fine_tokens, spatial_shape):
        """
        Build fine + coarse hierarchical visual tokens.

        Fine tokens:
            original visual tokens, e.g. 14 × 14 = 196 tokens.

        Coarse tokens:
            pooled visual tokens, e.g. 7 × 7 = 49 tokens.

        Args:
            fine_tokens: [B, N, D]
            spatial_shape: (H, W)

        Returns:
            hierarchical tokens: [B, N + N_c, D]
        """
        B, N, D = fine_tokens.shape

        if spatial_shape is None:
            H = int(math.sqrt(N))
            W = H
            if H * W != N:
                raise ValueError(
                    f"Cannot infer spatial shape from N={N}."
                )
        else:
            H, W = spatial_shape

        fine_2d = fine_tokens.transpose(1, 2).contiguous().view(B, D, H, W)

        coarse_h = max(1, H // 2)
        coarse_w = max(1, W // 2)

        coarse_2d = F.adaptive_avg_pool2d(
            fine_2d,
            output_size=(coarse_h, coarse_w)
        )

        coarse_tokens = coarse_2d.flatten(2).transpose(1, 2).contiguous()

        hierarchical_tokens = torch.cat(
            [fine_tokens, coarse_tokens],
            dim=1
        )

        return hierarchical_tokens

    def _original_branch(self, TextFeature, wordFeature, VisionFeature):
        """
        Original MSVA branch.

        This branch is preserved to reduce the risk of performance degradation.
        """
        vision1, vision2 = torch.chunk(VisionFeature, chunks=2, dim=2)

        text_embed = self.text_proj(TextFeature).unsqueeze(1)
        word_embed = self.word_proj(wordFeature).unsqueeze(1)

        vision1_norm = self.orig_norm_v1(vision1)
        text_norm = self.orig_norm_text(text_embed)

        attn_output1, attn_weights1 = self.text_to_vision(
            query=vision1_norm,
            key=text_norm,
            value=text_norm
        )

        vision1_refined = vision1 + self.dropout(attn_output1)

        vision2_norm = self.orig_norm_v2(vision2)
        word_norm = self.orig_norm_word(word_embed)

        attn_output2, attn_weights2 = self.word_to_vision(
            query=vision2_norm,
            key=word_norm,
            value=word_norm
        )

        vision2_refined = vision2 + self.dropout(attn_output2)

        vision1_pooled = vision1_refined.mean(dim=1)
        vision2_pooled = vision2_refined.mean(dim=1)

        combined = torch.cat(
            [vision1_pooled, vision2_pooled],
            dim=-1
        )

        gate = self.gate_net(combined)
        gated_feature = combined * gate

        normalized = self.norm3(gated_feature)
        ffn_output = self.ffn(normalized)

        output = gated_feature + self.dropout(ffn_output)

        return output, {
            "orig_text_attn": attn_weights1,
            "orig_word_attn": attn_weights2
        }

    def _enhanced_branch(self, TextFeature, wordFeature, VisionFeature, spatial_shape):
        """
        Enhanced MSVA branch.

        Text/word features are used as queries.
        Visual tokens are used as keys and values.
        """
        # Visual projection
        visual_tokens = self.enh_vision_proj(VisionFeature)  # [B, N, D]

        # Spatial-azimuth encoding
        visual_tokens = self.spatial_azimuth(
            visual_tokens,
            spatial_shape=spatial_shape
        )

        # Fine-grained visual tokens
        fine_tokens = self.enh_vision_norm(visual_tokens)

        # Fine + coarse hierarchical tokens
        hierarchical_tokens = self._build_hierarchical_tokens(
            visual_tokens,
            spatial_shape=spatial_shape
        )

        hierarchical_tokens = self.enh_vision_norm(hierarchical_tokens)

        # Text query: decoder hidden state h_t
        text_query = self.enh_text_query_proj(TextFeature).unsqueeze(1)
        text_query = self.enh_query_norm(text_query)

        # Word query: previous word embedding e(y_{t-1})
        word_query = self.enh_word_query_proj(wordFeature).unsqueeze(1)
        word_query = self.enh_query_norm(word_query)

        # Global text-guided attention over hierarchical visual tokens
        global_context, global_attn = self.global_text_to_vision(
            query=text_query,
            key=hierarchical_tokens,
            value=hierarchical_tokens
        )

        # Word-level attention over fine-grained visual tokens
        word_context, word_attn = self.word_semantic_to_vision(
            query=word_query,
            key=fine_tokens,
            value=fine_tokens
        )

        global_context = global_context.squeeze(1)
        word_context = word_context.squeeze(1)

        semantic_visual = torch.cat(
            [global_context, word_context],
            dim=-1
        )

        candidate = self.enh_out_proj(semantic_visual)

        visual_residual = self.enh_visual_residual(
            VisionFeature.mean(dim=1)
        )

        enh_gate_input = torch.cat(
            [
                candidate,
                visual_residual,
                TextFeature,
                wordFeature
            ],
            dim=-1
        )

        enh_gate = self.enh_gate_net(enh_gate_input)

        enhanced_output = enh_gate * candidate + (1.0 - enh_gate) * visual_residual

        enhanced_output = self.enh_norm1(enhanced_output)

        enhanced_output = enhanced_output + self.dropout(
            self.enh_ffn(self.enh_norm2(enhanced_output))
        )

        return enhanced_output, {
            "global_attn": global_attn,
            "word_attn": word_attn,
            "enh_gate": enh_gate
        }

    def forward(
            self,
            TextFeature,
            wordFeature,
            VisionFeature,
            spatial_shape=None,
            return_attn=False
    ):
        """
        Args:
            TextFeature:
                Global text feature, preferably decoder hidden state h_t.
                Shape: [B, text_dim]

            wordFeature:
                Word-level feature, preferably previous word embedding e(y_{t-1}).
                Shape: [B, embed_dim]

            VisionFeature:
                Visual feature from HDAF.
                Shape: [B, N, encoder_dim], [B, H, W, encoder_dim], or [B, encoder_dim, H, W]

            spatial_shape:
                Optional spatial shape (H, W). For 224 × 224 input images, usually (14, 14).

            return_attn:
                Whether to return attention maps and gates.

        Returns:
            output:
                [B, encoder_dim]
        """
        VisionFeature, inferred_shape = self._to_tokens(VisionFeature)

        if spatial_shape is None:
            spatial_shape = inferred_shape

        original_output, original_info = self._original_branch(
            TextFeature,
            wordFeature,
            VisionFeature
        )

        enhanced_output, enhanced_info = self._enhanced_branch(
            TextFeature,
            wordFeature,
            VisionFeature,
            spatial_shape=spatial_shape
        )

        residual_gate_input = torch.cat(
            [
                original_output,
                enhanced_output,
                TextFeature,
                wordFeature
            ],
            dim=-1
        )

        residual_gate = self.residual_gate(residual_gate_input)

        # Important:
        # gamma is initialized as 0, so the module initially behaves exactly like the original MSVA branch.
        output = original_output + self.gamma * residual_gate * self.dropout(enhanced_output)

        if return_attn:
            info = {}
            info.update(original_info)
            info.update(enhanced_info)
            info["residual_gate"] = residual_gate
            info["gamma"] = self.gamma.detach()

            return output, info

        return output

class TextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=1024, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.attention2 = TVAttention(encoder_dim, embed_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

        #self.decode_step = nn.LSTMCell(attention_dim+attention_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.top_down_attention = nn.LSTMCell(decoder_dim+encoder_dim+embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.language_attention = nn.LSTMCell(encoder_dim+decoder_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.textencoder = TextEncoder(input_size=embed_dim, hidden_size=decoder_dim, output_size=attention_dim)
        self.nnimg = nn.Linear(encoder_dim, attention_dim)


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # 64   64
        encoder_out = encoder_out[sort_ind]

        #64 196 2048
        encoded_captions = encoded_captions[sort_ind]
        #64 52
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        embeddings1 = embeddings.clone()
        text_feature = self.textencoder(embeddings1)


        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        encoder_out_mean = encoder_out.mean(1)
        encoder_out_mean1 = encoder_out_mean.clone()
        img_feature = self.nnimg(encoder_out_mean1).squeeze(1)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            '''
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h1[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            '''

            out_feature = self.attention2(h2[:batch_size_t],  embeddings[:batch_size_t, t, :], encoder_out[:batch_size_t])

            h1, c1 = self.top_down_attention(
                torch.cat([h2[:batch_size_t], out_feature, embeddings[:batch_size_t, t, :]], dim=1),
                (h1[:batch_size_t], c1[:batch_size_t]))  # (batch_size_t, decoder_dim)
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h1[:batch_size_t])
            h2, c2 = self.language_attention(
                torch.cat([h1[:batch_size_t], attention_weighted_encoding[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)

            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind, img_feature, text_feature
