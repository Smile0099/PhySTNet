import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 3
configs.batch_size = 3
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 60
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 20
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# data related
configs.input_dim = 6
configs.output_dim = 6
configs.input_length = 28
configs.output_length = 28
configs.input_gap = 1
configs.pred_shift = 24

# model related
configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim = (32, 48, 48, 32)
configs.d_attn = 32
configs.ssr_decay_rate = 0.8e-4
