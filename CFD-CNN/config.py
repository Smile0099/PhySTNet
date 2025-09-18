import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 1
configs.device = torch.device('cuda:1')
configs.batch_size_test = 8
configs.batch_size = 8
configs.lr = 0.0005
configs.weight_decay = 0
configs.display_interval = 60
configs.num_epochs = 100000
configs.early_stopping = True
configs.patience = 50
configs.gradient_clipping = True
configs.clipping_threshold = 1.

# data related
configs.input_dim = 6
configs.input_length = 28
configs.input_h = 42
configs.input_w = 42
configs.input_gap = 1
configs.pred_shift = 28

# model related
configs.in_shape = (28,6,42,42)
configs.hid_S = 32
configs.hid_T = 256
configs.N_S = 2
configs.N_T = 8
configs.incep_ker = [3,5,7,11]
configs.groups = 4

configs.model_type = 'msta'
configs.mlp_ratio = 8.
configs.drop = 0.0
configs.drop_path = 0.1
configs.spatio_kernel_enc = 3
configs.spatio_kernel_dec = 3
configs.act_inplace = True