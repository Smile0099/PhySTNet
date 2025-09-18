import numpy as np
from torch.utils.data import Dataset
from sa_convlstm import SAConvLSTM
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = configs.device
model = SAConvLSTM(configs.input_dim, configs.hidden_dim, configs.d_attn, configs.kernel_size).to(configs.device)
net = torch.load('checkpoint_135.33531856536865.chk')
model.load_state_dict(net['net'])
model.eval()
dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
with torch.no_grad():
    start_time = datetime.datetime.now()
    for j, (input_sst, nino_true) in enumerate(dataloader_eval):
        nino_pred = model(input_sst.float(), train=False)
        if j == 0:
            pred = nino_pred
            label = nino_true
        else:
            pred = torch.cat((pred, nino_pred), 0)
            label = torch.cat((label, nino_true), 0)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
np.savez('result.npz', pred=pred.cpu(), label=label.cpu())