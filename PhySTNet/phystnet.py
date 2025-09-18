import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from config import configs

class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(y)


class MultiScaleSpatialAttention(nn.Module):
    def __init__(self, kernel_sizes=(3, 7)):
        super().__init__()
        self.convs = nn.ModuleList()
        self.kernel_sizes = kernel_sizes
        for k in kernel_sizes:
            padding = k // 2
            self.convs.append(nn.Conv2d(2, 1, kernel_size=k, padding=padding, bias=False))

        self.alpha = nn.Parameter(torch.ones(len(kernel_sizes)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        maps = [conv(y) for conv in self.convs]

        weights = torch.softmax(self.alpha, dim=0)
        fused = sum(w * m for w, m in zip(weights, maps))
        return self.sigmoid(fused)


class GatedFusionAttention(nn.Module):
    def __init__(self, in_channels: int, eca_k_size: int = 3):
        super().__init__()
        self.ca = EfficientChannelAttention(in_channels, k_size=eca_k_size)
        self.sa = MultiScaleSpatialAttention()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca_map = self.ca(x)
        sa_map = self.sa(x)
        attn = ca_map * sa_map
        x_enh = x + x * attn
        g = self.gate(x)
        out = x + g * (x_enh - x)
        return out


class TemporalAttentionUnit(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int = 4,
                 growth_step: int = 4,
                 min_len: Optional[int] = 8,
                 max_len: Optional[int] = None):
        super().__init__()
        self.growth_step = max(1, growth_step)
        self.min_len = min_len
        self.max_len = max_len
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())

    @torch.no_grad()
    def _update_buffer(self, buf: List[torch.Tensor], token: torch.Tensor):
        buf.append(token)

    def _window_size(self, buf_len: int) -> int:
        win = buf_len // self.growth_step

        if win < self.min_len:
            return self.min_len

        if self.max_len is not None and win > self.max_len:
            return self.max_len

        return win

    def forward(self, h_cur, buf: List[torch.Tensor]):
        B, C, H, W = h_cur.shape
        q = F.adaptive_avg_pool2d(h_cur, 1).flatten(1).unsqueeze(1)

        if buf:
            win = self._window_size(len(buf))
            kv = torch.stack(buf[-win:], 1)
            out, _ = self.mha(q, kv, kv)
            alpha = self.proj(out.squeeze(1)).view(B, C, 1, 1)
            h_cur = h_cur * (1 + 0.5 * alpha)
        self._update_buffer(buf, q.squeeze(1).detach())
        return h_cur, buf


# ─────────────────── Spatio-Temporal LSTM Cell ───────────────────
class SpatioTemporalLSTMCellv2(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width,
                 filter_size, stride, layer_norm,
                 tau_growth_step: int = 4,
                 tau_min_len: Optional[int] = None,
                 tau_max_len: Optional[int] = None):                       # ← 修改
        super().__init__()
        self.num_hidden = num_hidden
        pad = filter_size // 2
        def _blk(inp, outp, k):
            seq = [nn.Conv2d(inp, outp, k, stride=stride, padding=pad, bias=False)]
            if layer_norm: seq.append(nn.LayerNorm([outp, height, width]))
            return nn.Sequential(*seq)

        self.conv_x = _blk(in_channel, num_hidden * 7, filter_size)
        self.conv_h = _blk(num_hidden, num_hidden * 4, filter_size)
        self.conv_m = _blk(num_hidden, num_hidden * 3, filter_size)
        self.conv_o = _blk(num_hidden * 2, num_hidden, filter_size)
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, 1, bias=False)

        self.tau = TemporalAttentionUnit(num_hidden,4, tau_growth_step, tau_min_len, tau_max_len)
        self.tau_buffer: List[torch.Tensor] = []                            # ← 修改
        self._forget_bias = 1.0

    def forward(self, x_t, h_t, c_t, m_t):
        x_cat = self.conv_x(x_t)
        h_cat = self.conv_h(h_t)
        m_cat = self.conv_m(m_t)

        i_x, f_x, g_x, i_xp, f_xp, g_xp, o_x = torch.split(x_cat, self.num_hidden, 1)
        i_h, f_h, g_h, o_h = torch.split(h_cat, self.num_hidden, 1)
        i_m, f_m, g_m = torch.split(m_cat, self.num_hidden, 1)

        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self._forget_bias)
        g = torch.tanh(g_x + g_h)
        delta_c = i * g
        c_new = f * c_t + delta_c

        i_p = torch.sigmoid(i_xp + i_m)
        f_p = torch.sigmoid(f_xp + f_m + self._forget_bias)
        g_p = torch.tanh(g_xp + g_m)
        delta_m = i_p * g_p
        m_new = f_p * m_t + delta_m

        mem = torch.cat([c_new, m_new], 1)
        o = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o * torch.tanh(self.conv_last(mem))

        h_new, self.tau_buffer = self.tau(h_new, self.tau_buffer)
        return h_new, c_new, m_new, delta_c, delta_m


class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.F_small = nn.Conv2d(input_dim, F_hidden_dim, kernel_size=3, padding=1)
        self.F_large = nn.Conv2d(input_dim, F_hidden_dim, kernel_size=7, padding=3)

        self.F_merge = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(F_hidden_dim * 2, input_dim, kernel_size=1),
            nn.GroupNorm(4, input_dim)
        )

        self.convgate = nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1, bias=bias)

        self.channel_attn = EfficientChannelAttention(input_dim)

    def forward(self, x, hidden):
        x = x * self.channel_attn(x)

        F1 = self.F_small(hidden)
        F2 = self.F_large(hidden)
        hidden_tilde = hidden + self.F_merge(torch.cat([F1, F2], dim=1))

        K = torch.sigmoid(self.convgate(torch.cat([x, hidden], dim=1)))
        return hidden_tilde + K * (x - hidden_tilde)


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super().__init__()
        self.H = []
        self.device = device
        self.cell_list = nn.ModuleList(
            [PhyCell_Cell(input_dim, F_hidden_dims[i], kernel_size) for i in range(n_layers)])

    def forward(self, x, first_timestep=False):
        if first_timestep:
            self.H = [torch.zeros(x.size(0), x.size(1), *x.shape[-2:],
                                  device=self.device) for _ in self.cell_list]
        for i, cell in enumerate(self.cell_list):
            self.H[i] = cell(x if i == 0 else self.H[i - 1], self.H[i])
        return self.H, self.H



class phystnet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        C, H, W = configs.in_shape
        self.configs = configs
        self.device = configs.device

        self.frame_channel = configs.patch_size ** 2 * C
        height, width = H // configs.patch_size, W // configs.patch_size
        self.h, self.w = height, width

        cell_list = []
        for i in range(configs.num_layers):
            in_ch = self.frame_channel if i == 0 else configs.num_hidden[i - 1]
            cell_list.append(SpatioTemporalLSTMCellv2(
                in_ch, configs.num_hidden[i], height, width,
                configs.filter_size, configs.stride, configs.layer_norm,
                tau_growth_step=configs.tau_growth_step,
                tau_min_len=configs.tau_min_len,
                tau_max_len=configs.tau_max_len
            ))
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(configs.num_hidden[-1] + 216,
                                   self.frame_channel, 1, bias=False)
        self.adapter = nn.Conv2d(configs.num_hidden[0], configs.num_hidden[0], 1, bias=False)
        self.phycell = PhyCell((7, 7), 216, [49], 1, (7, 7), self.device)

        self.W_1 = nn.Parameter(torch.randn(1, 216, 7, 7))
        self.W_2 = nn.Parameter(torch.randn(1, 256, 7, 7))
        self.CBAM_f = GatedFusionAttention(216)

    def forward(self, input, target,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):
        frames = torch.cat([input, target], 1).to(self.device)
        B, T, C, H, W = frames.shape

        frames = F.unfold(frames.view(B * T, C, H, W),
                          self.configs.patch_size, stride=self.configs.unfoldstride
                          ).view(B, T, C * self.configs.patch_size ** 2, self.h, self.w)

        for cell in self.cell_list:
            cell.tau_buffer.clear()

        h_t = [torch.zeros(B, nh, self.h, self.w, device=self.device)
               for nh in self.configs.num_hidden]
        c_t = [t.clone() for t in h_t]
        memory = torch.zeros_like(h_t[0])

        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:
                mask_true = torch.bernoulli(
                    scheduled_sampling_ratio * torch.ones(frames.size(0), self.configs.aft_seq_length - 1, 1, 1, 1)).to(
                    self.device)
            else:
                teacher_forcing = False
        else:
            teacher_forcing = False

        x_gen = frames[:, self.configs.pre_seq_length - 1].clone()
        next_frames = []

        for t in range(self.configs.total_length - 1):
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            elif not teacher_forcing:
                net = x_gen
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            net = self.CBAM_f(net)
            phy_input = net

            h_t[0], c_t[0], memory, delta_c, delta_m = \
                self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.configs.num_layers):
                h_t[i], c_t[i], memory, _, _ = \
                    self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            phy_out, _ = self.phycell(phy_input, first_timestep=(t == 0))
            phy_out = phy_out[-1] + phy_input

            x_gen = torch.cat([phy_out * self.W_1, h_t[-1] * self.W_2], 1)
            x_gen = self.conv_last(x_gen)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, 1)
        next_frames = F.fold(
            next_frames.view(-1, self.frame_channel, self.h * self.w),
            (H, W), self.configs.patch_size, stride=self.configs.unfoldstride
        ).view(B, self.configs.total_length - 1, C, H, W)

        if not train:
            next_frames = next_frames[:, -self.configs.aft_seq_length:]
        return next_frames

