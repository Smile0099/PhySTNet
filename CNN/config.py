import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 1
configs.device = torch.device('cuda:1')
configs.batch_size = 8
configs.batch_size_test = 8
configs.weight_decay = 0
configs.display_interval = 60
configs.num_epochs = 1000
configs.early_stopping = True
configs.patience = 50
configs.gradient_clipping = False
configs.clipping_threshold = 1.0

# data related
configs.input_dim = 6
configs.output_dim = 6
configs.input_length = int(1 * 28 / 1)
configs.output_length = int(1 * 28 / 1)
configs.input_gap = 1
configs.pred_shift = 28

# model
configs.d_model = 256
configs.lr = 5e-4
configs.ssr_decay_rate = 5e-5
configs.warmup = 5000
configs.n_future = 28
configs.cond_time = 28
configs.num_in_vars = 6
configs.patch_sizes = ((20, 20), (10, 10))
configs.overlaps = ((1, 1), (1, 1))
configs.modes = ((4, 4), (2, 2))
configs.img_size = (42, 42)
configs.lift_channel = 32
configs.dim = 64
configs.vit_depth = (3, 3)
configs.num_heads = 1
configs.mlp_dim = 64
configs.act = "gelu"
