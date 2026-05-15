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


class TVAttention(nn.Module):


    def __init__(self, encoder_dim, embed_dim, attention_dim, text_dim=1000):
        super(TVAttention, self).__init__()


        assert encoder_dim % 2 == 0, "encoder_dim必须是偶数"
        self.half_dim = encoder_dim // 2
        self.attention_dim = attention_dim


        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.GELU(),  # 比ReLU更平滑
            nn.Dropout(0.1)
        )


        self.word_proj = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.GELU()
        )


        self.vision_enhance = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim),
            nn.LayerNorm(self.half_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )


        self.text_align = nn.Linear(attention_dim, self.half_dim) if attention_dim != self.half_dim else nn.Identity()
        self.word_align = nn.Linear(attention_dim, self.half_dim) if attention_dim != self.half_dim else nn.Identity()


        self.text_guided_attn = nn.MultiheadAttention(
            embed_dim=self.half_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )


        self.word_guided_attn = nn.MultiheadAttention(
            embed_dim=self.half_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )


        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.half_dim, self.half_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.half_dim // 8, self.half_dim, 1),
            nn.Sigmoid()
        )


        self.fusion_gate = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),  # 输入原始+处理后的特征
            nn.LayerNorm(encoder_dim),
            nn.Sigmoid()
        )


        self.output = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.Dropout(0.1)
        )


        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, TextFeature, wordFeature, VisionFeature):
        """
        Forward propagation.
        :param TextFeature: text feature, tensor of dimension (batch_size, text_dim)
        :param wordFeature: word embedding, tensor of dimension (batch_size, embed_dim)
        :param VisionFeature: encoded images, tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: attention weighted encoding, tensor of dimension (batch_size, encoder_dim)
        """
        batch_size, num_pixels, _ = VisionFeature.shape


        vision1, vision2 = torch.chunk(VisionFeature, chunks=2, dim=2)
        # vision1, vision2: (batch, num_pixels, half_dim)


        text_proj = self.text_proj(TextFeature)  # (batch, attention_dim)
        text_query = text_proj.unsqueeze(1)  # (batch, 1, attention_dim)
        text_query_aligned = self.text_align(text_query)  # (batch, 1, half_dim)


        vision1_enhanced = self.vision_enhance(vision1)  # (batch, num_pixels, half_dim)


        vision1_attn, attn_weights = self.text_guided_attn(
            query=text_query_aligned,
            key=vision1_enhanced,
            value=vision1
        )  # (batch, 1, half_dim)


        vision1_mean = vision1.mean(dim=1)  # (batch, half_dim)
        vision1_out = 0.7 * vision1_mean + 0.3 * vision1_attn.squeeze(1)


        word_proj = self.word_proj(wordFeature).unsqueeze(1)  # (batch, 1, attention_dim)
        word_query_aligned = self.word_align(word_proj)  # (batch, 1, half_dim)


        channel_weight = self.channel_attn(vision2.transpose(1, 2)).transpose(1, 2)
        vision2_enhanced = vision2 * channel_weight  # (batch, num_pixels, half_dim)


        vision2_attn, _ = self.word_guided_attn(
            query=word_query_aligned,
            key=vision2_enhanced,
            value=vision2_enhanced
        )  # (batch, 1, half_dim)


        vision2_mean = vision2.mean(dim=1)  # (batch, half_dim)
        vision2_out = 0.6 * vision2_mean + 0.4 * vision2_attn.squeeze(1)


        vision_concat = torch.cat([vision1_out, vision2_out], dim=1)  # (batch, encoder_dim)


        vision_raw = VisionFeature.mean(dim=1)  # (batch, encoder_dim)
        gate_input = torch.cat([vision_concat, vision_raw], dim=1)  # (batch, encoder_dim * 2)
        gate = self.fusion_gate(gate_input)  # (batch, encoder_dim)

        vision_fused = gate * vision_concat + (1 - gate) * vision_raw


        output = self.output(vision_fused)  # (batch, encoder_dim)

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
