import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.star import STAR
from layers.star_block import STARBlock
from layers.patch_embed import PatchEmbed1D
# from modules.ms_patch import MSPatchEmbed
# from modules.ms_router import MSPatchRouter
# from modules.patch_filter import PatchFilter
# from modules.patch_mixer import PatchMixer
from model.modules.cac_block import CACBlock
from model.modules.gcac_block import GCACBlock
from model.modules.gstar_block import GStarBlock
from model.modules.mfh_nb import QuantileHead, quantile_losses, mfh_nb_fuse, MFHNBCalib
from model.modules.gated_fusion import GatedFusion
from model.modules.channel_mixer import ChannelMixerHead
from model.modules.frequency_enhance import FrequencyEnhance
from model.modules.tcn_embed import TCNEmbedPrefilter
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.gate_init_bias = float(getattr(configs, 'gate_init_bias', -2.0))
        self.use_gated_residual_flag = bool(getattr(configs, 'use_gated_residual', 0))
        self.gate_attn_enabled = bool(getattr(configs, 'gate_attn', 1))
        self.gate_ffn_enabled = bool(getattr(configs, 'gate_ffn', 1))
        gated_layers_spec = str(getattr(configs, 'gated_layers', '')).strip()
        if gated_layers_spec:
            layers_set = set()
            for tok in gated_layers_spec.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    layers_set.add(int(tok))
                except ValueError:
                    continue
            self.gated_layers = sorted(layers_set)
        else:
            self.gated_layers = None
        self.log_gate_stats = bool(getattr(configs, 'log_gate_stats', 0))
        self.gate_log_interval = int(getattr(configs, 'gate_log_interval', 200))
        
        # STAR and Patch Embedding parameters
        self.use_star = int(getattr(configs, 'use_star', 0))
        self.use_gstar = int(getattr(configs, 'use_gstar', 0))
        self.use_patch_embed = getattr(configs, 'use_patch_embed', 0)
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.patch_stride = getattr(configs, 'patch_stride', 8)
        self.patch_norm = getattr(configs, 'patch_norm', 'instance')
        self.patch_pos_emb = getattr(configs, 'patch_pos_emb', 'sincos')
        self.use_ci = getattr(configs, 'use_ci', 1)
        self.use_star_fusion_gate = bool(getattr(configs, 'use_star_fusion_gate', 1))
        self.gstar_gate_bias = float(getattr(configs, 'gstar_gate_bias', -2.0))
        self.gstar_star_bias = float(getattr(configs, 'gstar_star_bias', -4.0))
        self.gstar_use_ln = bool(getattr(configs, 'gstar_use_ln', 1))
        self.gstar_enable_star_gate = bool(getattr(configs, 'gstar_enable_star_gate', 1))
        self.gstar_enable_fusion_gate = bool(getattr(configs, 'gstar_enable_fusion_gate', 1))
        self.gstar_rms_norm = bool(getattr(configs, 'gstar_rms_norm', 0))
        self.gstar_cosine_mod = bool(getattr(configs, 'gstar_cosine_mod', 0))
        self.gstar_couple_tcn = bool(getattr(configs, 'gstar_couple_tcn', 0))
        self.use_cac_fusion_gate = bool(getattr(configs, 'use_cac_fusion_gate', 1))
        self.use_gcac = int(getattr(configs, 'use_gcac', 0))
        self.gcac_enable_pre = bool(getattr(configs, 'gcac_pre_gate', 1))
        self.gcac_enable_post = bool(getattr(configs, 'gcac_post_gate', 1))
        self.gcac_pre_bias = float(getattr(configs, 'gcac_pre_gate_bias', -1.0))
        self.gcac_post_bias = float(getattr(configs, 'gcac_post_gate_bias', -1.0))
        self.use_mfh_nb = int(getattr(configs, 'use_mfh_nb', 0))
        self.mfh_nb_lambda_q = float(getattr(configs, 'mfh_nb_lambda_q', 0.5))
        self.mfh_nb_lambda_mono = float(getattr(configs, 'mfh_nb_lambda_mono', 1e-3))
        self.mfh_nb_calib_path = str(getattr(configs, 'mfh_nb_calib_path', '') or '')
        self.mfh_nb_p_lo = float(getattr(configs, 'mfh_nb_p_lo', 20.0))
        self.mfh_nb_p_hi = float(getattr(configs, 'mfh_nb_p_hi', 80.0))
        self.mfh_nb_log_alpha = int(getattr(configs, 'mfh_nb_log_alpha', 0))
        self._skip_mfh_nb_fusion = False
        self.quantile_head = None
        self.mfh_nb_calib = None
        self._mfh_nb_quantiles = None
        self._mfh_nb_alpha = None
        self._mfh_nb_loss = None
        # -----------------------------------------------------------------
        # Legacy compatibility: older scripts still pass --use_gcac_gate /
        # --gcac_gate_bias to control the single post-fusion gate. We fold
        # those aliases into the new pre/post switch interface so both
        # generations of configs behave identically.
        legacy_gate_flag = getattr(configs, 'use_gcac_gate', None)
        if legacy_gate_flag is not None:
            self.gcac_enable_post = bool(legacy_gate_flag)
        legacy_gate_bias = getattr(configs, 'gcac_gate_bias', None)
        if legacy_gate_bias is not None:
            self.gcac_post_bias = float(legacy_gate_bias)
        # expose the legacy attribute names so that checkpoints / logs that
        # still touch them do not break at runtime.
        self.use_gcac_gate = self.gcac_enable_post
        self.gcac_gate_bias = self.gcac_post_bias
        # -----------------------------------------------------------------
        
        # Initialize STAR (pre) is removed; keep compatibility no-op
        self.star = None
            
        # Initialize Patch Embedding module
        if self.use_patch_embed:
            self.patch_embed = PatchEmbed1D(
                d_model=configs.d_model,
                patch_len=self.patch_len,
                patch_stride=self.patch_stride,
                norm=self.patch_norm,
                pos_emb=self.patch_pos_emb,
                use_ci=bool(self.use_ci)
            )
        else:
            self.patch_embed = None
            
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        encoder_layers = []
        for l in range(configs.e_layers):
            layer_gate_active = self.use_gated_residual_flag
            if layer_gate_active and self.gated_layers is not None:
                layer_gate_active = l in self.gated_layers
            encoder_layers.append(
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            use_temp=bool(getattr(configs, 'use_temp_attn', 0)),
                            tau=float(getattr(configs, 'attn_tau', 1.0)),
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_gated_residual=layer_gate_active,
                    gate_init_bias=self.gate_init_bias,
                    log_gate_stats=self.log_gate_stats,
                    gate_log_interval=self.gate_log_interval,
                    layer_id=l,
                    enable_attn_gate=self.gate_attn_enabled,
                    enable_ffn_gate=self.gate_ffn_enabled
                )
            )
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.use_mfh_nb:
            self.quantile_head = QuantileHead(configs.d_model, configs.pred_len, configs.c_out)

        # STARBlock variants (post-forecast)
        self.star_block = None
        self.star_fusion = None
        self.gstar_block = None
        self.star_kernel = None
        self.gstar_kernel = None
        star_c_out = int(getattr(configs, 'c_out', 1))
        star_kernel = int(getattr(configs, 'moving_avg', 25))

        if self.use_gstar:
            if self.use_star:
                print("[PS] Warning: both use_star and use_gstar requested. GStar will take precedence.")
            self.gstar_block = GStarBlock(
                c_out=star_c_out,
                k=star_kernel,
                star_gate_bias=self.gstar_star_bias,
                use_ln=self.gstar_use_ln,
                fusion_gate_bias=self.gstar_gate_bias,
                use_star_gate=self.gstar_enable_star_gate,
                use_fusion_gate=self.gstar_enable_fusion_gate,
                rms_norm=self.gstar_rms_norm,
                cosine_mod=self.gstar_cosine_mod,
                couple_tcn=self.gstar_couple_tcn,
            )
            self.gstar_kernel = star_kernel
        elif self.use_star:
            self.star_block = STARBlock(c_out=star_c_out, k=star_kernel, gate_bias=-4.0, use_ln=True)
            self.star_kernel = star_kernel
            if self.use_star_fusion_gate:
                self.star_fusion = GatedFusion(dim=star_c_out, gate_init_bias=self.gate_init_bias)

        # 单尺度 Patch 嵌入构造器
        def single_embed_ctor(P, S, **kwargs):
            return PatchEmbed1D(
                d_model=configs.d_model,
                patch_len=P,
                patch_stride=S,
                norm=getattr(configs, 'patch_norm', 'instance'),
                pos_emb=getattr(configs, 'patch_pos_emb', 'sincos'),
                use_ci=bool(getattr(configs, 'use_ci', 1))
            )

        # 多尺度 Patch 模块（暂时未使用，注释掉以避免导入错误）
        # self.ms_embed = None
        # self.ms_router = None
        # self.patch_filter = None
        # self.patch_mixer = None
        self._aux_loss = torch.zeros(1)

        # if getattr(configs, 'use_ms_patch', 0):
        #     patch_set = [int(x) for x in getattr(configs, 'ms_patch_set', '8,16,32').split(",")]
        #     stride_ratio = getattr(configs, 'ms_stride_ratio', 0.5)
        #     fuse = getattr(configs, 'ms_fuse', 'gate')
        #     self.ms_embed = MSPatchEmbed(single_embed_ctor, patch_set, stride_ratio, fuse=fuse)

        # if getattr(configs, 'use_ms_router', 0):
        #     patch_set = [int(x) for x in getattr(configs, 'ms_patch_set', '8,16,32').split(",")]
        #     stride_ratio = getattr(configs, 'ms_stride_ratio', 0.5)
        #     hidden = getattr(configs, 'router_hidden', 64)
        #     budget_weight = getattr(configs, 'router_budget', 0.0)
        #     self.ms_router = MSPatchRouter(single_embed_ctor, patch_set, stride_ratio,
        #                                    hidden=hidden, budget_weight=budget_weight)

        # if getattr(configs, 'use_patch_filter', 0):
        #     topk = getattr(configs, 'filter_topk', 8)
        #     season = getattr(configs, 'filter_season', '')
        #     self.patch_filter = PatchFilter(topk=topk, season=season)

        # if getattr(configs, 'use_patch_mixer', 0):
        #     groups = getattr(configs, 'mixer_groups', 8)
        #     dropout = getattr(configs, 'mixer_dropout', 0.1)
        #     self.patch_mixer = PatchMixer(d_model=configs.d_model, groups=groups, dropout=dropout)

        # CAC-Block: 时间轴周期归纳偏置
        self.cac = None
        self.gcac = None
        self.cac_fusion = None
        use_cac_flag = int(getattr(configs, 'use_cac', 0))
        if self.use_gcac:
            if use_cac_flag:
                print("[PS] Warning: both use_gcac and use_cac requested. GCAC will take precedence.")
            self.gcac = GCACBlock(
                d_model=configs.d_model,
                kernel_sizes=getattr(configs, 'cac_kernel_sizes', '3,5,7'),
                dilations=getattr(configs, 'cac_dilations', '1,2,4'),
                topk=getattr(configs, 'cac_topk', 5),
                dropout=getattr(configs, 'cac_dropout', 0.1),
                enable_pre_gate=self.gcac_enable_pre,
                enable_post_gate=self.gcac_enable_post,
                pre_gate_bias=self.gcac_pre_bias,
                post_gate_bias=self.gcac_post_bias,
            )
        elif use_cac_flag:
            self.cac = CACBlock(
                d_model=configs.d_model,
                kernel_sizes=getattr(configs, 'cac_kernel_sizes', '3,5,7'),
                dilations=getattr(configs, 'cac_dilations', '1,2,4'),
                topk=getattr(configs, 'cac_topk', 5),
                dropout=getattr(configs, 'cac_dropout', 0.1),
            )
            if self.use_cac_fusion_gate:
                self.cac_fusion = GatedFusion(dim=configs.d_model, gate_init_bias=self.gate_init_bias)
            # print(f"[PS] CAC-Block enabled.")  # 已在下方统一打印

        # =========================
        # [FEM] 频域增强模块（放在 Encoder 之前）
        # =========================
        self.use_freq_enhance = bool(getattr(configs, 'use_freq_enhance', 0))
        if self.use_freq_enhance:
            bands = int(getattr(configs, 'freq_bands', 3))
            freq_init = getattr(configs, 'freq_init', '')
            mix = float(getattr(configs, 'freq_mix', 0.5))
            if freq_init:
                init_tuple = tuple(float(x) for x in freq_init.split(','))
                assert len(init_tuple) == bands, "freq_init length must match freq_bands"
            else:
                init_tuple = (1.0, 1.0, 0.1) if bands == 3 else tuple([1.0 for _ in range(bands)])
            self.freq_enhance = FrequencyEnhance(
                seq_len=self.seq_len, n_bands=bands, init_weights=init_tuple, mix=mix
            )
            # print(f"[PS] FreqEnhance enabled (pre-encoder, L={self.seq_len}, F={self.seq_len//2+1}).")  # 已废弃
        else:
            self.freq_enhance = None

        # TCN-Embed Prefilter (channel-independent temporal conv)
        self.use_tcn_embed = bool(getattr(configs, 'use_tcn_embed', 0))
        self.tcn_stack = int(getattr(configs, 'tcn_stack', 1))
        # Legacy aliases for backward compatibility
        tcn_k1 = getattr(configs, 'tcn_k1', getattr(configs, 'tcn_kernel', 3))
        tcn_d1 = getattr(configs, 'tcn_d1', getattr(configs, 'tcn_dilation', 1))
        self.tcn_k1 = int(tcn_k1)
        self.tcn_d1 = int(tcn_d1)
        self.tcn_k2 = int(getattr(configs, 'tcn_k2', 5))
        self.tcn_d2 = int(getattr(configs, 'tcn_d2', 1))
        self.tcn_k3 = int(getattr(configs, 'tcn_k3', 7))
        self.tcn_d3 = int(getattr(configs, 'tcn_d3', 2))
        self.tcn_use_gate = bool(getattr(configs, 'tcn_gate', 1))
        if self.use_tcn_embed:
            self.tcn_embed = TCNEmbedPrefilter(
                seq_len=int(getattr(configs, 'seq_len', self.seq_len)),
                c_out=int(getattr(configs, 'enc_in', configs.enc_in)),
                kernel1=self.tcn_k1,
                dilation1=self.tcn_d1,
                kernel2=self.tcn_k2,
                dilation2=self.tcn_d2,
                kernel3=self.tcn_k3,
                dilation3=self.tcn_d3,
                stack=self.tcn_stack,
                use_gate=self.tcn_use_gate,
            )
        else:
            self.tcn_embed = None

        # ChannelMixer-Head: 变量维跨通道混合
        self.channel_mixer = None
        if getattr(configs, 'use_channel_mixer', 0):
            self.channel_mixer = ChannelMixerHead(
                c_out=configs.c_out,
                hidden_ratio=getattr(configs, 'cm_hidden_ratio', 2.0),
                dropout=getattr(configs, 'cm_dropout', 0.1),
            )
            # print("[PS] ChannelMixer-Head enabled.")  # 已废弃

        # Print module status (只显示当前使用的模块)
        print(f"[PS] STARBlock enabled={int(self.star_block is not None)}")
        if self.star_block is not None:
            print(f"[PS] STAR fusion gate enabled={int(self.star_fusion is not None)} (k={self.star_kernel})")
        if self.gstar_block is not None:
            print(
                f"[PS] GStar enabled=1 (k={self.gstar_kernel}, fusion_gate={int(self.gstar_enable_fusion_gate)}, "
                f"star_gate={int(self.gstar_enable_star_gate)}, fusion_bias={self.gstar_gate_bias}, "
                f"star_bias={self.gstar_star_bias}, ln={int(self.gstar_use_ln)})"
            )
            print(
                f"  ↳ guards: rms_norm={int(self.gstar_rms_norm)} "
                f"cosine_mod={int(self.gstar_cosine_mod)} "
                f"couple_tcn={int(self.gstar_couple_tcn)}"
            )
        print(f"[PS] TCN-Embed enabled={int(self.use_tcn_embed)}")
        if self.use_tcn_embed:
            k_vals = [self.tcn_k1, self.tcn_k2, self.tcn_k3]
            d_vals = [self.tcn_d1, self.tcn_d2, self.tcn_d3]
            k_str = ",".join(str(k_vals[i]) for i in range(min(self.tcn_stack, len(k_vals))))
            d_str = ",".join(str(d_vals[i]) for i in range(min(self.tcn_stack, len(d_vals))))
            print(
                f"  ↳ stack={self.tcn_stack} "
                f"k=[{k_str}] "
                f"d=[{d_str}] "
                f"gate={int(self.tcn_use_gate)} zero-phase=on dc=on pad=reflect"
            )
        gated_enabled = int(self.use_gated_residual_flag)
        if gated_enabled:
            layer_desc = "all" if self.gated_layers is None else ("none" if not self.gated_layers else ",".join(str(idx) for idx in self.gated_layers))
            print(f"[PS] Gated Residual enabled=1 (layers={layer_desc}, attn={int(self.gate_attn_enabled)}, ffn={int(self.gate_ffn_enabled)}, bias={self.gate_init_bias})")
        else:
            print(f"[PS] Gated Residual enabled=0")
        cac_kernel = getattr(configs, 'cac_kernel_sizes', '3,5,7')
        cac_dilation = getattr(configs, 'cac_dilations', '1,2,4')
        cac_topk = getattr(configs, 'cac_topk', 5)
        if self.use_gcac:
            print(f"[PS] GCAC enabled=1 pre={int(self.gcac_enable_pre)} post={int(self.gcac_enable_post)} "
                  f"pre_bias={self.gcac_pre_bias} post_bias={self.gcac_post_bias}")
            if use_cac_flag:
                print(f"[PS] CAC requested (ks={cac_kernel}, dil={cac_dilation}, topk={cac_topk}) but disabled because GCAC takes priority.")
        elif use_cac_flag:
            print(f"[PS] CAC enabled=1 fusion_gate={int(self.cac_fusion is not None)} "
                  f"ks={cac_kernel} dil={cac_dilation} topk={cac_topk}")
        else:
            print(f"[PS] CAC enabled=0")
        if self.log_gate_stats:
            print(f"[PS] Gate stats logging enabled every {self.gate_log_interval} steps.")
        # 已废弃的模块（不再显示）
        # print(f"[PS] patch={self.use_patch_embed}, P={self.patch_len}, S={self.patch_stride}, norm={self.patch_norm}, pos={self.patch_pos_emb}, CI={self.use_ci}")
        # print(f"[PS] FreqEnhance enabled={int(self.use_freq_enhance)} bands={getattr(configs, 'freq_bands', 3)} mix={getattr(configs, 'freq_mix', 0.5)}")
        # print(f"[PS] ChannelMixer enabled={getattr(configs, 'use_channel_mixer', 0)} ratio={getattr(configs, 'cm_hidden_ratio', 2.0)}")
        
        # ---- 递归设置所有 FullAttention 的 ASP-Lite 开关 ----
        # 注意：FullAttention 已在文件顶部导入，无需再次导入
        
        def _apply_asp_lite_flags(module: nn.Module):
            """递归设置所有 FullAttention 实例的 QK-Norm 和 Cosine 开关"""
            for m in module.modules():
                if isinstance(m, FullAttention):
                    # 互斥保护：如果同时开启，Cosine 优先
                    use_qk = int(getattr(configs, 'use_qk_norm', 0))
                    use_cos = int(getattr(configs, 'use_cosine_attn', 0))
                    
                    if use_qk and use_cos:
                        print("[PS] Warning: Both QK-Norm and Cosine requested. Cosine takes priority.")
                        use_qk = 0  # 关闭 QK-Norm
                    
                    m.use_qk_norm = bool(use_qk)
                    m.qk_eps = float(getattr(configs, 'qk_eps', 1e-6))
                    m.use_cosine_attn = bool(use_cos)
                    m.cos_scale = float(getattr(configs, 'cos_scale', 1.0))
                    m.use_temp = bool(getattr(configs, 'use_temp_attn', 0))
                    m.tau = float(getattr(configs, 'attn_tau', 1.0))
        
        _apply_asp_lite_flags(self)
        
        # ASP-Lite 状态打印
        asp_qk = int(getattr(configs, 'use_qk_norm', 0))
        asp_cos = int(getattr(configs, 'use_cosine_attn', 0))
        asp_temp = int(getattr(configs, 'use_temp_attn', 0))
        if asp_qk or asp_cos or asp_temp:
            tau_val = float(getattr(configs, 'attn_tau', 1.0))
            print(f"[PS] ASP-Lite: QK-Norm={asp_qk}, CosineAttn={asp_cos}, Temp={asp_temp} (tau={tau_val})")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self._mfh_nb_quantiles = None
        self._mfh_nb_alpha = None
        self._mfh_nb_loss = None
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.tcn_embed is not None:
            x_enc = self.tcn_embed(x_enc)
        g_tcn = None
        if self.tcn_embed is not None and getattr(self.tcn_embed, "use_gate", False):
            gate_vals = []
            if hasattr(self.tcn_embed, "alpha1"):
                gate_vals.append(torch.sigmoid(self.tcn_embed.alpha1).mean())
            if self.tcn_embed.stack > 1 and hasattr(self.tcn_embed, "alpha2"):
                gate_vals.append(torch.sigmoid(self.tcn_embed.alpha2).mean())
            if self.tcn_embed.stack > 2 and hasattr(self.tcn_embed, "alpha3"):
                gate_vals.append(torch.sigmoid(self.tcn_embed.alpha3).mean())
            if gate_vals:
                g_tcn = torch.stack(gate_vals).mean().detach()

        # (Removed old pre-STAR path)

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding with new patch modules
        attn_mask = None

        # 1) 产出 tokens（多尺度模块已注释，简化为单路径）
        # if self.ms_router is not None:
        #     enc_out = self.ms_router(x_enc, x_mark_enc)     # (B, L*, D)
        #     self._aux_loss = self.ms_router.get_aux_loss()
        # elif self.ms_embed is not None:
        #     enc_out = self.ms_embed(x_enc, x_mark_enc)
        #     self._aux_loss = enc_out.new_zeros(1)
        if self.patch_embed is not None:
            # Use Patch Embedding instead of DataEmbedding
            enc_tokens, _meta = self.patch_embed(x_enc)  # [B, C, D]
            self._aux_loss = enc_tokens.new_zeros(1)
        else:
            # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
            enc_tokens = self.enc_embedding(x_enc, x_mark_enc)
            self._aux_loss = enc_tokens.new_zeros(1)

        # 2) PatchMixer（已注释）
        # if self.patch_mixer is not None:
        #     enc_tokens = self.patch_mixer(enc_tokens)

        # 3) PatchFilter -> attention mask（已注释）
        # if self.patch_filter is not None:
        #     attn_mask = self.patch_filter(enc_tokens)  # (B,1,L,L)

        # -------- [FEM] 于 Encoder 之前（保留原始频率信息）----------
        if self.freq_enhance is not None:
            enc_tokens = self.freq_enhance(enc_tokens)  # [B, N, D]
        # ---------------------------------------------------------

        # 4) 送入 encoder
        encoder_core, attns = self.encoder(enc_tokens, attn_mask=attn_mask)

        # 5) CAC-Block: 在 encoder 输出后、投影前做周期增强并通过门控融合
        fused_encoder = encoder_core
        if self.gcac is not None:
            fused_encoder = self.gcac(encoder_core)
        elif self.cac is not None:
            cac_out = self.cac(encoder_core)
            fused_encoder = self.cac_fusion(encoder_core, cac_out) if self.cac_fusion is not None else cac_out

        if self.use_mfh_nb and self.quantile_head is not None:
            q25, q50, q75 = self.quantile_head(fused_encoder)
            q25, q50, q75 = q25[..., :N], q50[..., :N], q75[..., :N]
            self._mfh_nb_quantiles = (q25, q50, q75)

        # B N E -> B N S -> B S N
        dec_out = self.projector(fused_encoder).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # 6) ChannelMixer-Head: 在变量维做全局混合（在反归一化后，形状 [B, L, C]）
        if self.channel_mixer is not None:
            dec_out = self.channel_mixer(dec_out)  # [B, L, C]

        # Post-forecast STAR/GStar refinement + gated fusion
        if self.gstar_block is not None:
            dec_out = self.gstar_block(dec_out, g_tcn=g_tcn)
        elif self.star_block is not None:
            star_out = self.star_block(dec_out)
            dec_out = self.star_fusion(dec_out, star_out) if self.star_fusion is not None else star_out

        return dec_out, attns


    def get_aux_loss(self):
        base = getattr(self, "_aux_loss", 0.0)
        extra = self._mfh_nb_loss if isinstance(self._mfh_nb_loss, torch.Tensor) else None
        if isinstance(base, torch.Tensor):
            total = base
            if extra is not None:
                total = total + extra
            return total
        return extra if extra is not None else base

    def compute_mfh_nb_loss(self, target: torch.Tensor) -> torch.Tensor:
        if not self.use_mfh_nb or self._mfh_nb_quantiles is None:
            return target.new_zeros(())
        q25, q50, q75 = self._mfh_nb_quantiles
        loss = quantile_losses(
            q25,
            q50,
            q75,
            target,
            lambda_q=self.mfh_nb_lambda_q,
            lambda_mono=self.mfh_nb_lambda_mono,
        )
        self._mfh_nb_loss = loss
        return loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def forward_with_quantiles(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        *,
        fuse: bool = False,
        calib: MFHNBCalib | None = None,
        log_alpha: bool = False,
    ):
        """
        Forward helper that exposes MFH-NB quantiles and optional fusion.

        Returns:
            y_main: [B, pred_len, C]
            (q25, q50, q75): each [B, pred_len, C]
            y_out: fused output if fusion enabled; otherwise identical to y_main
            alpha: fusion weights tensor or None
        """
        dec_out, _ = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        quantiles = self._mfh_nb_quantiles
        if quantiles is None:
            raise RuntimeError("MFH-NB quantiles unavailable; ensure use_mfh_nb=1 during calibration.")

        y_main = dec_out[:, -self.pred_len:, :]
        q25, q50, q75 = [q[:, -self.pred_len:, :] for q in quantiles]
        assert q50.shape[-1] == y_main.shape[-1], (
            f"Quantile channels {q50.shape[-1]} do not match main output {y_main.shape[-1]}"
        )

        should_fuse = bool(fuse)
        has_calib = calib is not None and calib.p20 is not None and calib.p80 is not None
        if not should_fuse or not has_calib:
            if should_fuse and not has_calib:
                print("[MFH-NB] calib missing -> skip fusion (no alpha applied).")
            self._mfh_nb_alpha = None
            return y_main, (q25, q50, q75), y_main, None

        y_fused, alpha = mfh_nb_fuse(y_main, q25, q50, q75, calib.p20, calib.p80)
        self._mfh_nb_alpha = alpha

        if log_alpha or bool(self.mfh_nb_log_alpha):
            alpha_mean = alpha.mean().item()
            alpha_median = alpha.median().item()
            print(f"[MFH-NB] alpha_mean={alpha_mean:.4f} alpha_p50={alpha_median:.4f}")

        return y_main, (q25, q50, q75), y_fused, alpha
