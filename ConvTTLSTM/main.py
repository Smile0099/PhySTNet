import numpy as np
from torch.utils.data import Dataset
from convttlstm_net import ConvTTLSTMNet
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = configs.device
model = ConvTTLSTMNet(configs.order, configs.steps, configs.ranks, configs.kernel_size,configs.bias, configs.hidden_channels, configs.layers_per_block,configs.skip_stride, configs.input_dim, configs.output_dim).to(configs.device)

net = torch.load('checkpoint_84.29003524780273.chk')
model.load_state_dict(net['net'])
model.eval()
dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
with torch.no_grad():
    start_time = datetime.datetime.now()
    for j, (input_sst, nino_true) in enumerate(dataloader_eval):
        nino_pred = model(input_sst.float(),nino_true.float(), train=False)
        if j == 0:
            pred = nino_pred
            label = nino_true
        else:
            pred = torch.cat((pred, nino_pred), 0)
            label = torch.cat((label, nino_true), 0)
    end_time = datetime.datetime.now()
    print(end_time-start_time)
np.savez('result.npz', pred=pred.cpu(), label=label.cpu())
