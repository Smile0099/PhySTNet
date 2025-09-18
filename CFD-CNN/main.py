import numpy as np
from torch.utils.data import Dataset
from model import SimVP_Model
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau



device = configs.device
model = SimVP_Model(configs).to(configs.device)
net = torch.load('checkpoint_CFD_CNN')
model.load_state_dict(net['net'])
model.eval()
dataloader_eval = DataLoader(dataset_test, batch_size=8, shuffle=False)
with torch.no_grad():
    start_time = datetime.datetime.now()
    for j, (input_sst, nino_true) in enumerate(dataloader_eval):
        nino_pred = model(input_sst.float())
        if j == 0:
            pred = nino_pred
            label = nino_true
        else:
            pred = torch.cat((pred, nino_pred), 0)
            label = torch.cat((label, nino_true), 0)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
np.savez('result.npz', pred=pred.cpu(), label=label.cpu())