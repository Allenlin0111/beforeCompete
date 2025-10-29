from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np

from model.modules.mfh_nb import MFHNBCalib, quantile_losses

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._diag_ctx = None
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        try:
            if getattr(self.args, "tcn_alpha3_bias", None) is not None:
                if getattr(self.args, "tcn_stack", 0) >= 3 and hasattr(model_ref, "tcn_embed"):
                    import torch
                    if hasattr(model_ref.tcn_embed, "alpha3"):
                        with torch.no_grad():
                            model_ref.tcn_embed.alpha3.fill_(float(self.args.tcn_alpha3_bias))
                        print(f"[TCN] init alpha3 to bias={self.args.tcn_alpha3_bias}")
        except Exception as _e:
            print(f"[TCN] alpha3 bias init skipped: {_e}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _diag_is_active(self):
        return bool(int(getattr(self.args, 'diag_enable', 0)))

    def _diag_should_log_jsonl(self):
        return bool(int(getattr(self.args, 'diag_log_jsonl', 1)))

    def _diag_prepare(self, setting):
        horizon = getattr(self.args, 'pred_len', 1)
        weekly_period = max(1, int(getattr(self.args, 'diag_psd_weekly_period', 168)))
        rfft_bins = horizon // 2 + 1
        k_cut = max(1, min(rfft_bins, int(round(horizon / weekly_period)) or 1))

        out_root = Path(getattr(self.args, 'diag_outdir', './diagnostics/'))
        outdir = out_root / setting
        outdir.mkdir(parents=True, exist_ok=True)

        self._diag_ctx = {
            'setting': setting,
            'outdir': outdir,
            'train_path': outdir / 'diag_train.jsonl',
            'val_path': outdir / 'diag_val.jsonl',
            'latest_path': outdir / 'diag_latest.txt',
            'pending': 0,
            'k_cut': k_cut,
            'phase': None,
            'current_epoch': 0,
        }

    def _diag_collect_gates(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        gate_stats = {}

        def _register(name, tensor):
            if tensor is None:
                return
            det = tensor.detach()
            if det.numel() == 0:
                return
            gate_stats[name] = float(torch.sigmoid(det).mean().cpu().item())

        tcn = getattr(model, 'tcn_embed', None)
        if tcn is not None:
            for attr in ('alpha', 'alpha1', 'alpha2', 'alpha3'):
                if hasattr(tcn, attr):
                    _register(f'tcn_embed.{attr}', getattr(tcn, attr))
            for name, param in tcn.named_parameters(recurse=False):
                lname = name.lower()
                if 'alpha' in lname or 'gate' in lname:
                    _register(f'tcn_embed.{name}', param)

        for name, param in model.named_parameters():
            if 'alpha' in name and name not in gate_stats:
                _register(name, param)

        return gate_stats or None

    def _diag_extract_arrays(self, outputs, batch_y):
        f_dim = -1 if self.args.features == 'MS' else 0
        preds = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        return preds.detach().cpu().numpy(), target.detach().cpu().numpy()

    def _diag_compute_batch_stats(self, preds_np, trues_np, k_cut):
        residual = preds_np - trues_np
        abs_err = np.abs(residual)
        batch_size, horizon, channels = abs_err.shape

        step_sum = abs_err.sum(axis=(0, 2))
        step_den = np.full(horizon, batch_size * channels, dtype=np.float64)
        channel_sum = abs_err.sum(axis=(0, 1))
        channel_den = np.full(channels, batch_size * horizon, dtype=np.float64)
        overall_abs = abs_err.sum()
        overall_count = batch_size * horizon * channels

        channel_mae = np.divide(
            channel_sum,
            channel_den,
            out=np.zeros_like(channel_sum, dtype=np.float64),
            where=channel_den > 0,
        )
        total_channel = channel_mae.sum()
        if channels > 0 and total_channel > 0:
            top_k = max(1, int(np.ceil(0.05 * channels)))
            indices = np.argsort(channel_mae)[::-1][:top_k]
            top_share = float(channel_mae[indices].sum() / total_channel)
        else:
            top_share = 0.0

        fft_vals = np.fft.rfft(residual.reshape(-1, horizon), axis=1)
        power = np.abs(fft_vals) ** 2
        lowfreq_num = float(power[:, :k_cut].sum())
        lowfreq_den = float(power.sum())
        lowfreq_ratio = float(lowfreq_num / lowfreq_den) if lowfreq_den > 0 else 0.0

        step_mae = np.divide(
            step_sum,
            step_den,
            out=np.zeros_like(step_sum, dtype=np.float64),
            where=step_den > 0,
        )
        overall_mae = float(overall_abs / overall_count) if overall_count > 0 else 0.0

        return {
            'step_mae': step_mae,
            'channel_mae': channel_mae,
            'top_share': float(top_share),
            'lowfreq_ratio': lowfreq_ratio,
            'overall_mae': overall_mae,
            'raw': {
                'step_sum': step_sum,
                'step_den': step_den,
                'channel_sum': channel_sum,
                'channel_den': channel_den,
                'overall_abs': float(overall_abs),
                'overall_count': float(overall_count),
                'lowfreq_num': lowfreq_num,
                'lowfreq_den': lowfreq_den,
            },
        }

    def _diag_init_epoch_stats(self):
        horizon = getattr(self.args, 'pred_len', 1)
        return {
            'step_sum': np.zeros(horizon, dtype=np.float64),
            'step_den': np.zeros(horizon, dtype=np.float64),
            'channel_sum': None,
            'channel_den': None,
            'overall_abs': 0.0,
            'overall_count': 0.0,
            'lowfreq_num': 0.0,
            'lowfreq_den': 0.0,
            'gate_sums': {},
            'gate_counts': {},
        }

    def _diag_update_epoch_stats(self, epoch_stats, batch_stats):
        if epoch_stats is None or batch_stats is None:
            return
        raw = batch_stats['raw']
        epoch_stats['step_sum'] += raw['step_sum']
        epoch_stats['step_den'] += raw['step_den']
        if epoch_stats['channel_sum'] is None:
            epoch_stats['channel_sum'] = np.zeros_like(raw['channel_sum'], dtype=np.float64)
            epoch_stats['channel_den'] = np.zeros_like(raw['channel_den'], dtype=np.float64)
        epoch_stats['channel_sum'] += raw['channel_sum']
        epoch_stats['channel_den'] += raw['channel_den']
        epoch_stats['overall_abs'] += raw['overall_abs']
        epoch_stats['overall_count'] += raw['overall_count']
        epoch_stats['lowfreq_num'] += raw['lowfreq_num']
        epoch_stats['lowfreq_den'] += raw['lowfreq_den']

        gates = self._diag_collect_gates()
        if gates:
            for key, value in gates.items():
                epoch_stats['gate_sums'][key] = epoch_stats['gate_sums'].get(key, 0.0) + float(value)
                epoch_stats['gate_counts'][key] = epoch_stats['gate_counts'].get(key, 0) + 1

    def _diag_finalize_epoch_stats(self, epoch_stats):
        if epoch_stats is None or epoch_stats['overall_count'] <= 0:
            return None

        step_mae = np.divide(
            epoch_stats['step_sum'],
            np.maximum(epoch_stats['step_den'], 1e-9),
        )
        if epoch_stats['channel_sum'] is not None and epoch_stats['channel_den'] is not None:
            channel_mae = np.divide(
                epoch_stats['channel_sum'],
                np.maximum(epoch_stats['channel_den'], 1e-9),
            )
        else:
            channel_mae = np.array([], dtype=np.float64)

        if channel_mae.size > 0 and channel_mae.sum() > 0:
            top_k = max(1, int(np.ceil(0.05 * channel_mae.size)))
            indices = np.argsort(channel_mae)[::-1][:top_k]
            top_share = float(channel_mae[indices].sum() / channel_mae.sum())
        else:
            top_share = 0.0

        lowfreq_ratio = (
            float(epoch_stats['lowfreq_num'] / epoch_stats['lowfreq_den'])
            if epoch_stats['lowfreq_den'] > 0
            else 0.0
        )
        overall_mae = (
            float(epoch_stats['overall_abs'] / epoch_stats['overall_count'])
            if epoch_stats['overall_count'] > 0
            else 0.0
        )
        gates = None
        if epoch_stats['gate_sums']:
            gates = {
                key: epoch_stats['gate_sums'][key] / epoch_stats['gate_counts'][key]
                for key in epoch_stats['gate_sums']
            }

        return {
            'step_mae': step_mae,
            'channel_mae': channel_mae,
            'top_share': top_share,
            'lowfreq_ratio': lowfreq_ratio,
            'overall_mae': overall_mae,
            'gates': gates,
        }

    def _diag_write_jsonl(self, path, payload):
        if not self._diag_should_log_jsonl():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload) + '\n')
        latest = self._diag_ctx.get('latest_path') if self._diag_ctx else None
        if latest:
            latest.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def _diag_log_train_batch(self, setting, epoch, iter_idx, outputs, batch_y, batch_y_mark):
        if not self._diag_ctx:
            return
        preds_np, trues_np = self._diag_extract_arrays(outputs, batch_y)
        batch_stats = self._diag_compute_batch_stats(preds_np, trues_np, self._diag_ctx['k_cut'])

        payload = {
            'phase': 'train',
            'setting': setting,
            'epoch': int(epoch),
            'iter': int(iter_idx),
            'step_mae_head': [float(x) for x in batch_stats['step_mae'][:5]],
            'top5pct_channel_share': float(batch_stats['top_share']),
            'lowfreq_power_ratio': float(batch_stats['lowfreq_ratio']),
            'overall_mae': float(batch_stats['overall_mae']),
            'gate': self._diag_collect_gates(),
        }

        channel_mae = batch_stats['channel_mae']
        if channel_mae.size > 0:
            indices = np.argsort(channel_mae)[::-1][:min(5, channel_mae.size)]
            payload['channel_mae_top5'] = [
                {'channel': int(idx), 'mae': float(channel_mae[idx])} for idx in indices
            ]

        self._diag_write_jsonl(self._diag_ctx['train_path'], payload)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            diag_collect = self._diag_is_active() and self._diag_ctx and self._diag_ctx.get('phase') == 'val'
            epoch_stats = self._diag_init_epoch_stats() if diag_collect else None
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

                if diag_collect:
                    batch_stats = self._diag_compute_batch_stats(
                        pred.numpy(),
                        true.numpy(),
                        self._diag_ctx['k_cut'],
                    )
                    self._diag_update_epoch_stats(epoch_stats, batch_stats)

            if diag_collect:
                summary = self._diag_finalize_epoch_stats(epoch_stats)
                if summary:
                    payload = {
                        'phase': 'val',
                        'setting': self._diag_ctx['setting'],
                        'epoch': int(self._diag_ctx.get('current_epoch', 0)),
                        'step_mae_head': [float(x) for x in summary['step_mae'][:5]],
                        'top5pct_channel_share': float(summary['top_share']),
                        'lowfreq_power_ratio': float(summary['lowfreq_ratio']),
                        'overall_mae': float(summary['overall_mae']),
                        'gate': summary['gates'],
                    }
                    channel_mae = summary['channel_mae']
                    if channel_mae is not None and channel_mae.size > 0:
                        indices = np.argsort(channel_mae)[::-1][:min(5, channel_mae.size)]
                        payload['channel_mae_top5'] = [
                            {'channel': int(idx), 'mae': float(channel_mae[idx])} for idx in indices
                        ]
                    self._diag_write_jsonl(self._diag_ctx['val_path'], payload)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        diag_active = self._diag_is_active()
        if diag_active:
            self._diag_prepare(setting)
            diag_interval = int(getattr(self.args, 'diag_interval', 200))
            diag_sample_batches = max(1, int(getattr(self.args, 'diag_sample_batches', 1)))
        else:
            diag_interval = None
            diag_sample_batches = None

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        freeze3_epochs = int(getattr(self.args, "tcn_freeze3_epochs", 0) or 0)

        def _set_stage3_requires_grad(flag: bool):
            if getattr(self.args, "tcn_stack", 0) < 3:
                return
            if not hasattr(model_ref, "tcn_embed"):
                return
            for name, param in model_ref.named_parameters():
                if ".tcn_embed." in name and (".conv3" in name or "alpha3" in name or "norm3" in name or "bn3" in name):
                    param.requires_grad = flag

        for epoch in range(self.args.train_epochs):
            if getattr(self.args, "tcn_stack", 0) >= 3 and freeze3_epochs > 0:
                if epoch == 0:
                    _set_stage3_requires_grad(False)
                    print(f"[TCN] stage-3 frozen for first {freeze3_epochs} epoch(s)")
                elif epoch == freeze3_epochs:
                    _set_stage3_requires_grad(True)
                    print("[TCN] stage-3 unfrozen")

            try:
                import torch
                if hasattr(model_ref, "tcn_embed"):
                    tcn_mod = model_ref.tcn_embed
                    with torch.no_grad():
                        def _gate_mean(attr: str):
                            val = getattr(tcn_mod, attr, None)
                            if val is None:
                                return float('nan')
                            return torch.sigmoid(val).mean().item()
                        g1 = _gate_mean("alpha1")
                        g2 = _gate_mean("alpha2")
                        g3 = _gate_mean("alpha3") if getattr(self.args, "tcn_stack", 0) >= 3 else float('nan')
                    print(f"[TCN] gate mean: a1={g1:.3f}, a2={g2:.3f}, a3={g3:.3f}")
            except Exception:
                pass

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                def _forward_with_optional_quantiles():
                    if getattr(self.args, 'use_mfh_nb', 0):
                        y_main, quantiles, _, _ = self.model.forward_with_quantiles(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark, fuse=False
                        )
                        return y_main, quantiles
                    if self.args.output_attention:
                        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0], None
                    return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark), None

                def _compute_training_loss(outputs_tensor, quantiles_tuple):
                    f_dim = -1 if self.args.features == 'MS' else 0
                    preds = outputs_tensor[:, -self.args.pred_len:, f_dim:]
                    target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_val = criterion(preds, target)
                    if quantiles_tuple is not None:
                        q25, q50, q75 = quantiles_tuple
                        q25 = q25[:, -self.args.pred_len:, f_dim:]
                        q50 = q50[:, -self.args.pred_len:, f_dim:]
                        q75 = q75[:, -self.args.pred_len:, f_dim:]
                        q_loss = quantile_losses(
                            q25,
                            q50,
                            q75,
                            target,
                            lambda_q=getattr(self.args, 'mfh_nb_lambda_q', 0.5),
                            lambda_mono=getattr(self.args, 'mfh_nb_lambda_mono', 1e-3),
                        )
                        loss_val = loss_val + q_loss
                    if hasattr(self.model, "get_aux_loss"):
                        aux_val = self.model.get_aux_loss()
                        if isinstance(aux_val, torch.Tensor) and aux_val.numel() > 0:
                            loss_val = loss_val + aux_val
                    return loss_val

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, quantiles = _forward_with_optional_quantiles()
                        loss = _compute_training_loss(outputs, quantiles)
                else:
                    outputs, quantiles = _forward_with_optional_quantiles()
                    loss = _compute_training_loss(outputs, quantiles)

                train_loss.append(loss.item())

                if diag_active and self._diag_ctx:
                    trigger = (diag_interval is None or diag_interval <= 0)
                    if not trigger and diag_interval:
                        trigger = (i % diag_interval == 0)
                    if trigger:
                        self._diag_ctx['pending'] = diag_sample_batches
                    if self._diag_ctx.get('pending', 0) > 0:
                        with torch.no_grad():
                            self._diag_log_train_batch(
                                setting,
                                epoch + 1,
                                i + 1,
                                outputs,
                                batch_y,
                                batch_y_mark,
                            )
                        self._diag_ctx['pending'] -= 1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if diag_active and self._diag_ctx:
                self._diag_ctx['phase'] = 'val'
                self._diag_ctx['current_epoch'] = epoch + 1
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if diag_active and self._diag_ctx:
                self._diag_ctx['phase'] = 'test'
            test_loss = self.vali(test_data, test_loader, criterion)
            if diag_active and self._diag_ctx:
                self._diag_ctx['phase'] = None

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        calib = MFHNBCalib(getattr(self.args, 'mfh_nb_calib_path', '')) if getattr(self.args, 'use_mfh_nb', 0) else None
        if getattr(self.args, 'use_mfh_nb', 0) and (calib is None or calib.p20 is None or calib.p80 is None):
            print("[MFH-NB] WARN: mfh_nb_calib_path missing or invalid -> skip fusion (no alpha).")
        log_alpha = bool(getattr(self.args, 'mfh_nb_log_alpha', 0))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                def _forward_eval():
                    if getattr(self.args, 'use_mfh_nb', 0):
                        _, _, fused, _ = self.model.forward_with_quantiles(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            fuse=True,
                            calib=calib,
                            log_alpha=log_alpha,
                        )
                        return fused
                    if self.args.output_attention:
                        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = _forward_eval()
                else:
                    outputs = _forward_eval()

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        calib = MFHNBCalib(getattr(self.args, 'mfh_nb_calib_path', '')) if getattr(self.args, 'use_mfh_nb', 0) else None
        if getattr(self.args, 'use_mfh_nb', 0) and (calib is None or calib.p20 is None or calib.p80 is None):
            print("[MFH-NB] WARN: mfh_nb_calib_path missing or invalid -> skip fusion (no alpha).")
        log_alpha = bool(getattr(self.args, 'mfh_nb_log_alpha', 0))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                def _forward_pred():
                    if getattr(self.args, 'use_mfh_nb', 0):
                        _, _, fused, _ = self.model.forward_with_quantiles(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                            fuse=True,
                            calib=calib,
                            log_alpha=log_alpha,
                        )
                        return fused
                    if self.args.output_attention:
                        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = _forward_pred()
                else:
                    outputs = _forward_pred()
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
