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
configs.lr = 0.0005
configs.weight_decay = 0
configs.display_interval = 60
configs.num_epochs = 1000
configs.patience = 20
configs.reg_epoch = 50
configs.early_stopping = True
configs.gradient_clipping = False
configs.clipping_threshold = 1.


# data related
configs.pre_seq_length = 28
configs.aft_seq_length = 28
configs.total_length = configs.pre_seq_length + configs.aft_seq_length

# model
configs.ssr_decay_rate = 1.e-4
configs.in_shape = (6,42,42)
configs.num_layers = 4
configs.num_hidden = [256,256,256,256]
configs.sr_size = 2
configs.tau = 5
configs.cell_mode = 'normal'
configs.model_mode = 'normal'

configs.filter_size = 5
configs.patch_size = 6
configs.unfoldstride = configs.patch_size
configs.stride = 1
configs.layer_norm = 0
configs.tau_growth_step = 6
configs.tau_min_len = 8
configs.tau_max_len = 8