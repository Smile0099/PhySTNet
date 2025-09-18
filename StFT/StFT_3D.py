import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from einops import rearrange
from model_utils import TransformerLayer, get_2d_sincos_pos_embed
from config import configs


class StFTBlcok(nn.Module):
    def __init__(
        self,
        cond_time,
        freq_in_channels,
        in_dim,
        out_dim,
        out_channel,
        num_patches,
        modes,
        lift_channel=32,
        dim=256,
        depth=2,
        num_heads=1,
        mlp_dim=256,
        act="relu",
        grid_size=(4, 4),
        layer_indx=0,
    ):
        super(StFTBlcok, self).__init__()
        self.layer_indx = layer_indx
        self.cond_time = cond_time
        self.freq_in_channels = freq_in_channels
        self.modes = modes
        self.out_channel = out_channel
        self.lift_channel = lift_channel
        self.token_embed = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        self.pos_embed_fno = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        pos_embed_fno = get_2d_sincos_pos_embed(self.pos_embed_fno.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_fno.data.copy_(
            torch.from_numpy(pos_embed_fno).float().unsqueeze(0)
        )
        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.encoder_layers_fno = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.p = nn.Linear(freq_in_channels, lift_channel)
        self.linear = nn.Linear(
            modes[0] * modes[1] * (self.cond_time + self.layer_indx) * lift_channel * 2,
            dim,
        )
        self.q = nn.Linear(dim, modes[0] * modes[1] * 1 * lift_channel * 2)
        self.down = nn.Linear(lift_channel, out_channel)

    def forward(self, x):
        x_copy = x
        n, l, _, ph, pw = x.shape
        x_or = x[:, :, : self.cond_time * self.freq_in_channels]
        x_added = x[:, :, (self.cond_time * self.freq_in_channels) :]
        x_or = rearrange(
            x_or,
            "n l (t v) ph pw -> n l ph pw t v",
            t=self.cond_time,
            v=self.freq_in_channels,
        )
        grid_dup = x_or[:, :, :, :, :1, -2:].repeat(1, 1, 1, 1, self.layer_indx, 1)
        x_added = rearrange(
            x_added,
            "n l (t v) ph pw -> n l ph pw t v",
            t=self.layer_indx,
            v=self.freq_in_channels - 2,
        )
        x_added = torch.cat((x_added, grid_dup), axis=-1)
        x = torch.cat((x_or, x_added), axis=-2)
        x = self.p(x)
        x = rearrange(x, "n l ph pw t v -> (n l) v t ph pw")
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])[
            :, :, :, : self.modes[0], : self.modes[1]
        ]
        x_ft_real = (x_ft.real).flatten(1)
        x_ft_imag = (x_ft.imag).flatten(1)
        x_ft_real = rearrange(x_ft_real, "(n l) D -> n l D", n=n, l=l)
        x_ft_imag = rearrange(x_ft_imag, "(n l) D -> n l D", n=n, l=l)
        x_ft_real_imag = torch.cat((x_ft_real, x_ft_imag), axis=-1)
        x = self.linear(x_ft_real_imag)
        x = x + self.pos_embed_fno
        for layer in self.encoder_layers_fno:
            x = layer(x)
        x_real, x_imag = self.q(x).split(
            self.modes[0] * self.modes[1] * self.lift_channel, dim=-1
        )
        x_real = x_real.reshape(n * l, -1, 1, self.modes[0], self.modes[1])
        x_imag = x_imag.reshape(n * l, -1, 1, self.modes[0], self.modes[1])
        x_complex = torch.complex(x_real, x_imag)
        out_ft = torch.zeros(
            n * l,
            self.lift_channel,
            1,
            ph,
            pw // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :, : self.modes[0], : self.modes[1]] = x_complex
        x = torch.fft.irfftn(out_ft, s=(1, ph, pw))
        x = rearrange(x, "(n l) v t ph pw -> (n l) ph pw (v t)", n=n, l=l, t=1)
        x = self.down(x)
        x_fno = rearrange(x, "(n l) ph pw c -> n l c ph pw", n=n, l=l)
        x = x_copy
        _, _, _, ph, pw = x.shape
        x = x.flatten(2)
        x = self.token_embed(x) + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.head(x)
        x = rearrange(
            x, "n l (c ph pw) -> n l c ph pw", c=self.out_channel, ph=ph, pw=pw
        )
        x = x + x_fno
        return x



class StFT(nn.Module):
    def __init__(
        self,
        cond_time,
        num_vars,
        patch_sizes,
        overlaps,
        in_channels,
        out_channels,
        modes,
        img_size=(50, 50),
        lift_channel=32,
        dim=128,
        vit_depth=3,
        num_heads=1,
        mlp_dim=128,
        act="relu",
    ):
        super(StFT, self).__init__()

        blocks = []
        self.cond_time = cond_time
        self.num_vars = num_vars
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps
        for depth, (p1, p2) in enumerate(patch_sizes):
            H, W = img_size
            cur_modes = modes[depth]
            cur_depth = vit_depth[depth]
            overlap_h, overlap_w = overlaps[depth]

            step_h = p1 - overlap_h
            step_w = p2 - overlap_w

            pad_h = (step_h - (H - p1) % step_h) % step_h
            pad_w = (step_w - (W - p2) % step_w) % step_w
            H_pad = H + pad_h
            W_pad = W + pad_w

            num_patches_h = (H_pad - p1) // step_h + 1
            num_patches_w = (W_pad - p2) // step_w + 1

            num_patches = num_patches_h * num_patches_w
            if depth == 0:
                blocks.append(
                    StFTBlcok(
                        cond_time,
                        num_vars,
                        p1 * p2 * in_channels,
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        cur_modes,
                        lift_channel=lift_channel,
                        dim=dim,
                        depth=cur_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        act=act,
                        grid_size=(num_patches_h, num_patches_w),
                        layer_indx=depth,
                    )
                )
            else:
                blocks.append(
                    StFTBlcok(
                        cond_time,
                        num_vars,
                        p1 * p2 * (in_channels + out_channels),
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        cur_modes,
                        lift_channel=lift_channel,
                        dim=dim,
                        depth=cur_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        act=act,
                        grid_size=(num_patches_h, num_patches_w),
                        layer_indx=1,
                    )
                )

        self.blocks = nn.ModuleList(blocks)

    def create_normalized_grid(self,x):
        """
        根据输入 x 的形状自动创建一个归一化的空间位置 grid。

        输入:
            x: Tensor of shape [B, T, C, H, W]
        输出:
            grid: Tensor of shape [B, T, 2, H, W]，2 表示 (x, y) 归一化坐标
        """
        B, T, _, H, W = x.shape
        device = x.device

        # 创建归一化坐标：x ∈ [0, 1]、y ∈ [0, 1]
        x_coords = torch.linspace(0, 1, W, device=device)
        y_coords = torch.linspace(0, 1, H, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H, W]

        # 拼接成 (2, H, W)，其中 [0,:,:] 是 x， [1,:,:] 是 y
        grid = torch.stack([xx, yy], dim=0)  # [2, H, W]

        # 扩展到 [B, T, 2, H, W]
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, T, 2, H, W)

        return grid

    def forward(self, x, grid):
        grid_dup = self.create_normalized_grid(x)
        x = torch.cat((x, grid_dup), axis=2)
        x = rearrange(x, "B L C H W -> B (L C) H W")
        layer_outputs = []
        patches = x
        restore_params = []
        or_patches = x
        if True:
            for depth in range(len(self.patch_sizes)):
                if True:
                    p1, p2 = self.patch_sizes[depth]
                    overlap_h, overlap_w = self.overlaps[depth]

                    step_h = p1 - overlap_h
                    step_w = p2 - overlap_w

                    pad_h = (step_h - (patches.shape[2] - p1) % step_h) % step_h
                    pad_w = (step_w - (patches.shape[3] - p2) % step_w) % step_w
                    padding = (
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    )

                    patches = F.pad(patches, padding, mode="constant", value=0)
                    _, _, H_pad, W_pad = patches.shape

                    h = (H_pad - p1) // step_h + 1
                    w = (W_pad - p2) // step_w + 1

                    restore_params.append(
                        (p1, p2, step_h, step_w, padding, H_pad, W_pad, h, w)
                    )

                    patches = patches.unfold(2, p1, step_h).unfold(3, p2, step_w)
                    patches = rearrange(patches, "n c h w ph pw -> n (h w) c ph pw")

                    processed_patches = self.blocks[depth](patches)

                    patches = rearrange(
                        processed_patches, "n (h w) c ph pw -> n c h w ph pw", h=h, w=w
                    )

                    output = F.fold(
                        rearrange(patches, "n c h w ph pw -> n (c ph pw) (h w)"),
                        output_size=(H_pad, W_pad),
                        kernel_size=(p1, p2),
                        stride=(step_h, step_w),
                    )

                    overlap_count = F.fold(
                        rearrange(
                            torch.ones_like(patches),
                            "n c h w ph pw -> n (c ph pw) (h w)",
                        ),
                        output_size=(H_pad, W_pad),
                        kernel_size=(p1, p2),
                        stride=(step_h, step_w),
                    )
                    output = output / overlap_count
                    output = output[
                        :,
                        :,
                        padding[2] : H_pad - padding[3],
                        padding[0] : W_pad - padding[1],
                    ]
                    layer_outputs.append(output)
                    added = output
                    patches = torch.cat((or_patches, added.detach().clone()), axis=1)

        return layer_outputs




class StFTForecaster(nn.Module):
    """
    多步自回归 / Teacher-Forcing / Scheduled-Sampling 预测器
    ------------------------------------------------------
    forward(
        x_hist,                 # [B, k, V, H, W]
        n_future=None,          # 预测步长；默认用 self.n_future
        tgt_future=None,        # 训练时的教师真值 [B, n_future, V, H, W]
        train: bool = True,
        ssr_ratio: float = 0.0, # 0=全教师；1=全模型；(0,1)=随机
        heights: Tuple[int,int]|None = None,  # 裁剪 H 维
        src_mask=None, memory_mask=None       # 若以后要用 attention mask
    ) -> y_seq  [B, n_future, V, H', W]
    """

    # ---------- 1️⃣  构造 ----------
    def __init__(self, configs):
        super().__init__()
        self.n_future   = configs.n_future
        cond_time       = configs.cond_time
        num_in_vars     = configs.num_in_vars     # V
        patch_sizes     = configs.patch_sizes
        overlaps        = configs.overlaps
        modes           = configs.modes
        img_size        = configs.img_size
        lift_channel    = configs.lift_channel
        dim             = configs.dim
        vit_depth       = configs.vit_depth
        num_heads       = configs.num_heads
        mlp_dim         = configs.mlp_dim
        act             = configs.act
        self.device     = torch.device(configs.device)

        # 派生通道数
        num_vars    = num_in_vars + 2                    # V + (lon,lat)
        in_channels = num_vars * cond_time
        out_channels = num_in_vars                      # = V

        self.stft = StFT(
            cond_time    = cond_time,
            num_vars     = num_vars,
            patch_sizes  = patch_sizes,
            overlaps     = overlaps,
            in_channels  = in_channels,
            out_channels = out_channels,
            modes        = modes,
            img_size     = img_size,
            lift_channel = lift_channel,
            dim          = dim,
            vit_depth    = vit_depth,
            num_heads    = num_heads,
            mlp_dim      = mlp_dim,
            act          = act,
        ).to(self.device)

    # ---------- 2️⃣  前向 ----------
    def forward(
        self,
        x_hist,                      # [B,k,V,H,W]
        tgt_future,    # [B,n_future,V,H,W]
        train = True,
        ssr_ratio = 0.0,
    ) -> torch.Tensor:
        """
        训练:  需传入 tgt_future,  可设 ssr_ratio>0
        推理:  train=False, tgt_future=None  (完全自回归)
        """
        B, k, V, H, W = x_hist.shape
        assert k == self.stft.cond_time, "历史帧数必须等于 cond_time"

        n_future = self.n_future
        if train:
            assert tgt_future is not None and tgt_future.size(1) == n_future, \
                "训练模式必须提供与 n_future 等长的教师序列"

        x_hist = x_hist.to(self.device)
        preds: List[torch.Tensor] = []

        # ---------- 训练 / 验证 ----------
        if train:
            for t in range(n_future):
                y_next = self.stft(x_hist, grid=None)[-1]     # 单步输出 [B,V,H,W]
                preds.append(y_next)

                # ---- Scheduled Sampling ----
                teacher = tgt_future[:, t].to(self.device)    # 真值
                if ssr_ratio == 0.0:
                    next_in = teacher                         # 纯教师
                elif ssr_ratio == 1.0:
                    next_in = y_next.detach()                 # 纯模型
                else:
                    mask = torch.bernoulli(
                        torch.full((B, 1, 1, 1), ssr_ratio, device=self.device)
                    )
                    next_in = mask * teacher + (1 - mask) * y_next.detach()

                # 更新窗口
                x_hist = torch.cat([x_hist[:, 1:], next_in.unsqueeze(1)], dim=1)

        # ---------- 推理 ----------
        else:
            with torch.no_grad():
                for _ in range(n_future):
                    y_next = self.stft(x_hist, grid=None)[-1]
                    preds.append(y_next)
                    x_hist = torch.cat([x_hist[:, 1:], y_next.unsqueeze(1)], dim=1)

        y_seq = torch.stack(preds, dim=1)           # [B,n_future,V,H,W]

        return y_seq


if __name__ == "__main__":


    # 1. 定义核心超参
    model = StFTForecaster(configs).to(configs.device)

    # 2. 准备历史帧 [B, cond_time, V, H, W]
    x_hist = torch.randn(3, 24, 4, 40, 40).to(configs.device)
    y_future = torch.randn(3, 24, 4, 40, 40).to(configs.device)

    model.train()
    pred_train = model(
        x_hist,
        tgt_future=y_future,
        train=True,
        ssr_ratio=0.3,
    )
    print("=== Train ===")
    print("Input  :", x_hist.shape)  # [2,3,4,20,20]
    print("Target :", y_future.shape)  # [2,6,4,20,20]
    print("Output :", pred_train.shape)  # [2,6,4,20,20]

    # ---------------- 5. 推理模式 (自回归 n_future=6) ----------------
    model.eval()
    with torch.no_grad():
        pred_inf = model(x_hist, None, train=False)  # n_future 默认 6
    print("\n=== Inference ===")
    print("Output :", pred_inf.shape)  # [2,6,4,20,20]
