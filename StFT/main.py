import numpy as np
import xr
from torch.utils.data import Dataset
from StFT_3D import StFTForecaster
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime


device = configs.device
model = StFTForecaster(configs).to(device)
net = torch.load('checkpoint_100.00003242492676.chk')
model.load_state_dict(net['net'])
model.eval()

data = DataLoader(dataset_test, batch_size=3, shuffle=False)

with torch.no_grad():
    starttime = datetime.datetime.now()
    for i, (input, target) in enumerate(data):
        pred_temp = model(input.float().to(device), None, train=False)
        if i == 0:
            pred = pred_temp
            label = target
        else:
            pred = torch.cat((pred, pred_temp), 0)
            label = torch.cat((label, target), 0)
    endtime=datetime.datetime.now()
    print('SPEND TIME:',(endtime-starttime))

np.savez('result.npz', pred=pred.cpu(), label=label.cpu())