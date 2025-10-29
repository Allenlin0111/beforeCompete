import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='iTransformer')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='MTSF',
                        help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)') # See Figure 8 of our paper for the detail
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # STARBlock switch only
    ps = parser.add_argument_group('iTransformer_PS::STAR')
    ps.add_argument('--use_star', type=int, default=0, help='Enable STARBlock (0/1)')
    ps.add_argument('--use_gstar', type=int, default=0, help='Enable GStar (STAR + configurable gates) (0/1)')
    ps.add_argument('--gstar_gate_bias', type=float, default=-2.0, help='Init bias for the GStar fusion gate')
    ps.add_argument('--gstar_star_bias', type=float, default=-4.0, help='Init bias for the internal STAR gate used by GStar')
    ps.add_argument('--gstar_use_ln', type=int, default=1, help='Apply LayerNorm inside GStar (0/1)')
    ps.add_argument('--gstar_enable_star_gate', type=int, default=1, help='Enable STAR residual gate inside GStar (0/1)')
    ps.add_argument('--gstar_enable_fusion_gate', type=int, default=1, help='Enable fusion gate in GStar wrapper (0/1)')
    ps.add_argument('--gstar_rms_norm', type=int, default=0,
                    help='[GStar] Normalize (star - dec) by batch RMS before fusion (0/1)')
    ps.add_argument('--gstar_cosine_mod', type=int, default=0,
                    help='[GStar] Modulate alpha by cosine(dec, star) in [0,1] (0/1)')
    ps.add_argument('--gstar_couple_tcn', type=int, default=0,
                    help='[GStar] Couple alpha with TCN gate: alpha *= (1 - sigmoid(gamma)) (0/1)')
    ps.add_argument('--use_mfh_nb', type=int, default=0, help='Enable MFH-NB median fusion head (0/1)')
    ps.add_argument('--mfh_nb_lambda_q', type=float, default=0.5, help='Quantile loss weight for MFH-NB head')
    ps.add_argument('--mfh_nb_lambda_mono', type=float, default=1e-3, help='Monotonicity regulariser weight for MFH-NB')
    ps.add_argument('--mfh_nb_calib_path', type=str, default='', help='Path to MFH-NB calibration json (eval only)')
    ps.add_argument('--mfh_nb_p_lo', type=int, default=20, help='Lower percentile used when calibration is missing')
    ps.add_argument('--mfh_nb_p_hi', type=int, default=80, help='Upper percentile used when calibration is missing')
    ps.add_argument('--mfh_nb_log_alpha', type=int, default=0, help='Log MFH-NB alpha statistics during eval (0/1)')

    # Patch Embedding parameters
    parser.add_argument('--use_patch_embed', type=int, default=0, help='enable PatchTST-style embedding (0/1)')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length P')
    parser.add_argument('--patch_stride', type=int, default=8, help='patch stride S')
    parser.add_argument('--patch_norm', type=str, default='instance', choices=['none','layer','instance'], help='patch normalization')
    parser.add_argument('--patch_pos_emb', type=str, default='sincos', choices=['none','sincos'], help='patch position embedding')
    parser.add_argument('--use_ci', type=int, default=1, help='channel-independent projection (0/1)')

    # --- 多尺度 patch ---
    parser.add_argument("--use_ms_patch", type=int, default=0, help='enable multi-scale patch embedding (0/1)')
    parser.add_argument("--ms_patch_set", type=str, default="8,16,32", help='comma-separated patch sizes')
    parser.add_argument("--ms_stride_ratio", type=float, default=0.5, help='stride = P * ratio for each scale')
    parser.add_argument("--ms_fuse", type=str, default="gate", choices=["gate","concat"], help='multi-scale fusion method')

    # --- 自适应路由（基于多尺度）---
    parser.add_argument("--use_ms_router", type=int, default=0, help='enable multi-scale router (0/1)')
    parser.add_argument("--router_hidden", type=int, default=64, help='router hidden dimension')
    parser.add_argument("--router_budget", type=float, default=0.0, help='budget regularization weight')

    # --- PatchFilter 稀疏注意力 ---
    parser.add_argument("--use_patch_filter", type=int, default=0, help='enable patch filter (0/1)')
    parser.add_argument("--filter_topk", type=int, default=8, help='top-k connections per query')
    parser.add_argument("--filter_season", type=str, default="", help='seasonal offsets, e.g. "24,168"')

    # --- PatchMixer 通道混合 ---
    parser.add_argument("--use_patch_mixer", type=int, default=0, help='enable patch mixer (0/1)')
    parser.add_argument("--mixer_groups", type=int, default=8, help='mixer groups')
    parser.add_argument("--mixer_dropout", type=float, default=0.1, help='mixer dropout rate')

    # --- CAC-Block: 时间轴周期归纳偏置 ---
    parser.add_argument('--use_cac', type=int, default=0, help='enable CAC-Block (0/1)')
    parser.add_argument('--cac_kernel_sizes', type=str, default='3,5,7', help='comma-separated kernel sizes for multi-scale convolutions')
    parser.add_argument('--cac_dilations', type=str, default='1,2,4', help='comma-separated dilation rates')
    parser.add_argument('--cac_topk', type=int, default=5, help='top-k lags for auto-correlation')
    parser.add_argument('--cac_dropout', type=float, default=0.1, help='CAC-Block dropout rate')

    # --- ChannelMixer-Head: 变量维跨通道混合 ---
    parser.add_argument('--use_channel_mixer', type=int, default=0, help='enable ChannelMixer-Head (0/1)')
    parser.add_argument('--cm_hidden_ratio', type=float, default=2.0, help='hidden dimension ratio for channel mixer')
    parser.add_argument('--cm_dropout', type=float, default=0.1, help='ChannelMixer dropout rate')

    # --- TCN-Embed Prefilter (channel-independent temporal conv) ---
    parser.add_argument('--use_tcn_embed', type=int, default=0, help='enable TCN prefilter before series projection (0/1)')
    parser.add_argument('--tcn_stack', type=int, default=1, help='number of depthwise TCN stages (1 or 2)')
    parser.add_argument('--tcn_k1', type=int, default=3, help='kernel size for first TCN stage (odd number)')
    parser.add_argument('--tcn_d1', type=int, default=1, help='dilation factor for first TCN stage')
    parser.add_argument('--tcn_k2', type=int, default=5, help='kernel size for second TCN stage (odd number)')
    parser.add_argument('--tcn_d2', type=int, default=1, help='dilation factor for second TCN stage')
    parser.add_argument('--tcn_k3', type=int, default=7, help='kernel size for third TCN stage (odd number)')
    parser.add_argument('--tcn_d3', type=int, default=2, help='dilation factor for third TCN stage')
    parser.add_argument('--tcn_gate', type=int, default=1, help='enable scalar gates on TCN stages (0/1)')
    # === TCN stage-3 controls (safe no-op when tcn_stack < 3) ===
    parser.add_argument("--tcn_alpha3_bias", type=float, default=None,
                        help="If set and tcn_stack>=3, initialize alpha3 to this bias (e.g., -5.0) to delay stage-3 entry.")
    parser.add_argument("--tcn_freeze3_epochs", type=int, default=0,
                        help="If >0 and tcn_stack>=3, freeze all stage-3 params for the first N epochs, then unfreeze.")
    parser.add_argument('--tcn_kernel', dest='tcn_k1', type=int, default=argparse.SUPPRESS,
                        help='[legacy] alias for --tcn_k1')
    parser.add_argument('--tcn_dilation', dest='tcn_d1', type=int, default=argparse.SUPPRESS,
                        help='[legacy] alias for --tcn_d1')

    # --- Talking-Heads (post) switches ---
    parser.add_argument('--use_talking_heads', type=int, default=0, help='enable Talking-Heads (0/1)')
    parser.add_argument('--th_post', type=int, default=1, help='apply post-mixing (0/1)')
    parser.add_argument('--th_dropout', type=float, default=0.0, help='Talking-Heads dropout rate')  # 等价模式默认 0

    # ---- Temperature Softmax (pre-softmax temperature for attention) ----
    parser.add_argument('--use_temp_attn', type=int, default=0,
                        help='Enable temperature on attention logits before softmax (1=on, 0=off).')
    parser.add_argument('--attn_tau', type=float, default=1.0,
                        help='Temperature τ for attention logits (scores / τ). τ=1.0 means no change.')

    # ---- [FEM] 频域增强开关与参数 ----
    parser.add_argument('--use_freq_enhance', type=int, default=0, help='enable frequency enhancement module (FEM)')
    parser.add_argument('--freq_bands', type=int, default=3, help='number of frequency bands (e.g., 3 for low/mid/high)')
    parser.add_argument('--freq_init', type=str, default='', help='comma-separated init weights, length=freq_bands; empty->defaults')
    parser.add_argument('--freq_mix', type=float, default=0.5, help='residual mix coefficient in [0,1]')

    # ---- [Gateformer] Gated Residual Connections ----
    parser.add_argument('--use_gated_residual', type=int, default=0,
                        help='Enable Gateformer-inspired gated residual connections (0=off, 1=on)')
    parser.add_argument('--gate_init_bias', type=float, default=-2.0,
                        help='Bias init for gated residual/fusion (negative keeps identity, e.g. -2≈0.12)')
    parser.add_argument('--gated_layers', type=str, default='',
                        help='Comma-separated encoder layer ids (0-based) to enable gated residual; empty means all layers.')
    parser.add_argument('--gate_attn', type=int, default=1,
                        help='Enable gate on attention branch (1=on, 0=off)')
    parser.add_argument('--gate_ffn', type=int, default=1,
                        help='Enable gate on FFN branch (1=on, 0=off)')
    parser.add_argument('--use_star_fusion_gate', type=int, default=1,
                        help='Enable gated fusion after STAR block (1=on, 0=off)')
    parser.add_argument('--use_cac_fusion_gate', type=int, default=1,
                        help='Enable gated fusion after CAC block (1=on, 0=off)')
    parser.add_argument('--use_gcac', type=int, default=0,
                        help='Use GCAC module (CAC with internal gates) instead of plain CAC (1=on, 0=off)')
    parser.add_argument('--gcac_pre_gate', type=int, default=1,
                        help='Enable internal pre-gate within GCAC (1=on, 0=off)')
    parser.add_argument('--gcac_post_gate', type=int, default=1,
                        help='Enable internal post-gate within GCAC (1=on, 0=off)')
    parser.add_argument('--gcac_pre_gate_bias', type=float, default=-1.0,
                        help='Bias init for GCAC pre-gate (delta residual)')
    parser.add_argument('--gcac_post_gate_bias', type=float, default=-1.0,
                        help='Bias init for GCAC post-gate (fusion)')
    parser.add_argument('--log_gate_stats', type=int, default=0,
                        help='Print gate statistics during training (1=on, 0=off)')
    parser.add_argument('--gate_log_interval', type=int, default=200,
                        help='How many forward calls between gate stat prints (<=0 to print every step)')

    # --- Diagnostics logging ---
    parser.add_argument('--diag_enable', type=int, default=0,
                        help='enable lightweight diagnostics logging during training (0/1)')
    parser.add_argument('--diag_interval', type=int, default=200,
                        help='training iterations between diagnostic samples')
    parser.add_argument('--diag_sample_batches', type=int, default=2,
                        help='diagnostic batches to sample per trigger')
    parser.add_argument('--diag_psd_weekly_period', type=int, default=168,
                        help='weekly period threshold (in horizon steps) for PSD split')
    parser.add_argument('--diag_outdir', type=str, default='./diagnostics/',
                        help='output directory for diagnostic artefacts')
    parser.add_argument('--diag_log_jsonl', type=int, default=1,
                        help='write diagnostics to JSONL files (0/1)')

    # ---- [ASP-Lite] QK-Norm & Cosine Attention (分步消融) ----
    parser.add_argument('--use_qk_norm', type=int, default=0,
                        help='Enable QK-Norm: standardize Q,K along head dim')
    parser.add_argument('--qk_eps', type=float, default=1e-6,
                        help='Epsilon for QK-Norm variance clamp')
    parser.add_argument('--use_cosine_attn', type=int, default=0,
                        help='Enable Cosine Attention: L2-normalize Q,K, use cosine similarity')
    parser.add_argument('--cos_scale', type=float, default=1.0,
                        help='Scale/temperature for cosine attention logits')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # print('Args in experiment:')  # 参数过多，注释掉
    # print(args)
    
    # 只打印当前使用的关键模块状态
    if str(args.model).lower() == 'itransformer_ps':
        print("\n" + "="*60)
        print("iTransformer-PS 模块状态:")
        print("-"*60)
        use_star = int(getattr(args, 'use_star', 0))
        use_gstar = int(getattr(args, 'use_gstar', 0))
        use_cac_requested = int(getattr(args, 'use_cac', 0))
        use_gcac = int(getattr(args, 'use_gcac', 0))
        # GCAC reuses CAC internals; once it is enabled we treat CAC as inactive
        # to avoid confusing console output.
        use_cac_effective = int(use_cac_requested and not use_gcac)

        print(f"STAR Block:      {use_star}")
        if use_star:
            print(f"  ↳ STAR fusion gate: {getattr(args, 'use_star_fusion_gate', 1)}")
        print(f"GStar Block:     {use_gstar}")
        if use_gstar:
            print(f"  ↳ STAR gate:        {getattr(args, 'gstar_enable_star_gate', 1)} (bias={getattr(args, 'gstar_star_bias', -4.0)})")
            print(f"  ↳ Fusion gate:      {getattr(args, 'gstar_enable_fusion_gate', 1)} (bias={getattr(args, 'gstar_gate_bias', -2.0)})")
            print(f"  ↳ use_ln:           {getattr(args, 'gstar_use_ln', 1)}")
            print(f"  ↳ guards: rms_norm={getattr(args, 'gstar_rms_norm', 0)} "
                  f"cosine_mod={getattr(args, 'gstar_cosine_mod', 0)} "
                  f"couple_tcn={getattr(args, 'gstar_couple_tcn', 0)}")
        use_tcn_embed = int(getattr(args, 'use_tcn_embed', 0))
        print(f"TCN-Embed:       {use_tcn_embed}")
        if use_tcn_embed:
            stack = int(getattr(args, 'tcn_stack', 1))
            k_vals = [getattr(args, 'tcn_k1', 3), getattr(args, 'tcn_k2', 5), getattr(args, 'tcn_k3', 7)]
            d_vals = [getattr(args, 'tcn_d1', 1), getattr(args, 'tcn_d2', 1), getattr(args, 'tcn_d3', 2)]
            k_str = ",".join(str(k_vals[i]) for i in range(min(stack, len(k_vals))))
            d_str = ",".join(str(d_vals[i]) for i in range(min(stack, len(d_vals))))
            print(
                f"  ↳ stack={stack} "
                f"k=[{k_str}] "
                f"d=[{d_str}] "
                f"gate={getattr(args, 'tcn_gate', 1)} zero-phase=on dc=on pad=reflect"
            )
        if use_star and use_gstar:
            print("  ↳ Warning: both STAR and GStar requested; GStar will be used.")
        use_mfh_nb = int(getattr(args, 'use_mfh_nb', 0))
        print(f"MFH-NB Head:     {use_mfh_nb}")
        if use_mfh_nb:
            print(f"  ↳ lambda_q:        {getattr(args, 'mfh_nb_lambda_q', 0.5)} mono={getattr(args, 'mfh_nb_lambda_mono', 1e-3)}")
            calib_path = getattr(args, 'mfh_nb_calib_path', '')
            print(f"  ↳ calib_path:      {calib_path or 'N/A'} (p_lo={getattr(args, 'mfh_nb_p_lo', 20.0)}, p_hi={getattr(args, 'mfh_nb_p_hi', 80.0)})")

        print(f"CAC Block:       {use_cac_effective}")
        if use_cac_requested and use_gcac:
            print("  ↳ (requested, but disabled because GCAC takes priority)")
        elif use_cac_effective:
            print(f"  ↳ CAC fusion gate:  {getattr(args, 'use_cac_fusion_gate', 1)}")

        print(f"GCAC module:     {use_gcac}")
        if use_gcac:
            print(f"  ↳ pre_gate: {getattr(args, 'gcac_pre_gate', 1)} bias={getattr(args, 'gcac_pre_gate_bias', -1.0)}")
            print(f"  ↳ post_gate:{getattr(args, 'gcac_post_gate', 1)} bias={getattr(args, 'gcac_post_gate_bias', -1.0)}")
        gated = getattr(args, 'use_gated_residual', 0)
        if gated:
            layers = str(getattr(args, 'gated_layers', '')).strip()
            layer_desc = "all" if not layers else layers
            print(f"Gated Residual:  enabled (layers={layer_desc}, attn={getattr(args, 'gate_attn', 1)}, "
                  f"ffn={getattr(args, 'gate_ffn', 1)}, bias={getattr(args, 'gate_init_bias', -2.0)})")
        else:
            print(f"Gated Residual:  {gated}")
        asp_qk = int(getattr(args, 'use_qk_norm', 0))
        asp_cos = int(getattr(args, 'use_cosine_attn', 0))
        if asp_qk or asp_cos:
            print(f"ASP-Lite:        QK-Norm={asp_qk}, Cosine={asp_cos}")
        print("="*60 + "\n")
        
        # 已废弃的模块（不再显示）
        # print(f"[PS] patch={args.use_patch_embed}, P={args.patch_len}, S={args.patch_stride}, norm={args.patch_norm}, pos={args.patch_pos_emb}, CI={args.use_ci}")
        # print(f"[PS] TalkingHeads enabled={int(getattr(args, 'use_talking_heads', 0))} post={int(getattr(args, 'th_post', 1))} p={float(getattr(args, 'th_dropout', 0.0))}")
        # print(f"[PS] Temperature Softmax enabled={int(getattr(args, 'use_temp_attn', 0))} tau={float(getattr(args, 'attn_tau', 1.0))}")
        # print(f"[PS] FreqEnhance enabled={int(getattr(args, 'use_freq_enhance', 0))} bands={int(getattr(args, 'freq_bands', 3))} init=\"{getattr(args, 'freq_init', '')}\" mix={float(getattr(args, 'freq_mix', 0.5))}")

    if args.exp_name == 'partial_train': # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    else: # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast

    if args.is_training == 0 and int(getattr(args, 'use_mfh_nb', 0)) == 1:
        calib_path = getattr(args, 'mfh_nb_calib_path', '')
        if calib_path:
            print(f"[MFH-NB] calib: {calib_path}")
        else:
            print("[MFH-NB] WARN: no calib path given, fusion will be skipped.")


    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
