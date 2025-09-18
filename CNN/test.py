import numpy as np
import xr
from torch.utils.data import Dataset
from CNN import SimpleCNN
import torch
import torch.nn as nn
from config import configs
from torch.utils.data import DataLoader
import pickle
import math
import datetime

class NoamOpt:
    """
    learning rate warmup and decay
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class AutomaticWeightedLoss(nn.Module):
    """
    automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    Use method：
        model = Model()
        awl = AutomaticWeightedLoss(2)
        loss_1 = ...
        loss_2 = ...
        # learnable parameters
        optimizer = optim.Adam([{'params': model.parameters()},{'params': awl.parameters(), 'weight_decay': 0}])
        for i in range(epoch):
            for data, label1, label2 in data_loader:
            # forward
            pred1, pred2 = Model(data)
            # calculate losses
            loss1 = loss_1(pred1, label1)
            loss2 = loss_2(pred2, label2)
            # weigh losses
            loss_sum = awl(loss1, loss2)
            # backward
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
    """

    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)  # parameters的封装使得变量可以容易访问到

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        # +1 avoids the problem of log 0. The log sigma part has little impact on the overall loss
        return loss_sum

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.input_dim = configs.input_dim
        torch.manual_seed(5)
        self.network = SimpleCNN(configs).to(configs.device)
        self.awl = AutomaticWeightedLoss(self.input_dim)
        adam = torch.optim.Adam([{'params': self.network.parameters()}, {'params': self.awl.parameters()}], lr=0, weight_decay=configs.weight_decay)
        factor = math.sqrt(configs.d_model*configs.warmup)*0.0008
        self.opt = NoamOpt(configs.d_model, factor, warmup=configs.warmup, optimizer=adam)
        self.z500, self.t850, self.t2m, self.new, self.new1 , self.new2  = 'z500', 't850', 't2m', 'new', 'new1', 'new2'

    def loss(self, y_pred, y_true,idx):
        if idx == 'z500':
            idx = 0
        if idx == 't850':
            idx = 1
        if idx == 't2m':
            idx = 2
        if idx == 'new':
            idx = 3
        if idx == 'new1':
            idx = 4
        if idx == 'new2':
            idx = 5
        rmse = torch.mean((y_pred[:, :, idx] - y_true[:, :, idx])**2, dim=[2, 3])
        rmse = torch.sum(rmse.sqrt().mean(dim=0))
        return rmse

    def train_once(self, input_sst, sst_true, ssr_ratio):
        sst_pred = self.network(input_sst.float().to(self.device),sst_true.float().to(self.device),train=True, ssr_ratio=ssr_ratio)
        self.opt.optimizer.zero_grad()
        loss_z500 = self.loss(sst_pred, sst_true.float().to(self.device), self.z500)
        loss_t850 = self.loss(sst_pred, sst_true.float().to(self.device), self.t850)
        loss_t2m = self.loss(sst_pred, sst_true.float().to(self.device), self.t2m)
        loss_new = self.loss(sst_pred, sst_true.float().to(self.device), self.new)
        loss_new1 = self.loss(sst_pred, sst_true.float().to(self.device), self.new1)
        loss_new2 = self.loss(sst_pred, sst_true.float().to(self.device), self.new2)
        loss = self.awl(loss_z500, loss_t850, loss_t2m, loss_new, loss_new1, loss_new2)
        loss.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.opt.step()
        return loss_z500.item(), loss_t850.item(), loss_t2m.item(), loss_new.item(), loss_new1.item(),loss_new2.item(),loss

    def test(self, dataloader_test):
        sst_pred = []
        with torch.no_grad():
            for input_sst, sst_true in dataloader_test:
                sst = self.network(input_sst.float().to(self.device), None, train=False)
                sst_pred.append(sst)

        return torch.cat(sst_pred, dim=0)

    def infer(self, dataset, dataloader):
        self.network.eval()
        with torch.no_grad():
            sst_pred = self.test(dataloader)
            sst_true = torch.from_numpy(dataset.target).float().to(self.device)
            loss_z500 = self.loss(sst_pred, sst_true, self.z500).item()
            loss_t850 = self.loss(sst_pred, sst_true, self.t850).item()
            loss_t2m = self.loss(sst_pred, sst_true, self.t2m).item()
            loss_new = self.loss(sst_pred, sst_true, self.new).item()
            loss_new1 = self.loss(sst_pred, sst_true, self.new1).item()
            loss_new2 = self.loss(sst_pred, sst_true, self.new2).item()

        return loss_z500, loss_t850, loss_t2m, loss_new, loss_new1, loss_new2

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        ssr_ratio = 1
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()
            for j, (input_sst, sst_true) in enumerate(dataloader_train):
                if ssr_ratio > 0:
                    ssr_ratio = max(ssr_ratio - self.configs.ssr_decay_rate, 0)
                loss_z500, loss_t850, loss_t2m, loss_new, loss_new1, loss_new2, loss  = self.train_once(input_sst, sst_true, ssr_ratio)  # y_pred for one batch

                if (j+1) % self.configs.display_interval == 0:
                    print('batch training loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, ssr: {:.5f}, lr: {:.5f}, lr: {:.5f}, lr: {:.5f}'.format(loss_z500, loss_t850, loss_t2m, loss_new, loss_new1,loss_new2,loss, ssr_ratio, self.opt.rate()))

                # increase the number of evaluations in order not to miss the optimal point
                # which is feasible because of the less training time of ticsformer
                if (i+1 >= 6) and (j+1)%(self.configs.display_interval * 2) == 0:
                    loss_z500_eval_0, loss_t850_eval_0, loss_t2m_eval_0, loss_new1_eval_0, loss_new2_eval_0, loss_new_eval_0 = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
                    loss_eval_0 = loss_z500_eval_0 + loss_t850_eval_0 + loss_t2m_eval_0 + loss_new1_eval_0 + loss_new2_eval_0 + loss_new_eval_0
                    print('batch eval loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(loss_z500_eval_0, loss_t850_eval_0, loss_t2m_eval_0, loss_new_eval_0, loss_new1_eval_0, loss_new2_eval_0, loss_eval_0))
                    # self.save_model(chk_path + '_' + str(i) + '_' + str(loss_eval_0) + '.chk')
                    if loss_eval_0 < best:
                        self.save_model(chk_path + '_' + str(loss_eval_0) + '.chk')
                        best = loss_eval_0
                        count = 0

            # evaluation
            loss_z500_eval, loss_t850_eval, loss_t2m_eval, loss_new1_eval, loss_new2_eval, loss_new_eval = self.infer(dataset=dataset_eval, dataloader=dataloader_eval)
            loss_eval = loss_z500_eval + loss_t850_eval + loss_t2m_eval + loss_new1_eval + loss_new2_eval + loss_new_eval
            print('epoch eval loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(loss_z500_eval, loss_t850_eval, loss_t2m_eval, loss_new1_eval, loss_new2_eval, loss_new_eval, loss_eval))
            # self.save_model(chk_path + '_' + str(i) + '_' + str(loss_eval) + '.chk')
            if loss_eval >= best:
                count += 1
                print('eval loss is not reduced for {} epoch'.format(count))
            else:
                count = 0
                print('eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model(chk_path + '_' + str(loss_eval) + '.chk')
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, best score is {:5f}'.format(best))
                break

    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.opt.optimizer.state_dict()}, path)

def data_standardization(dataset):
    data = []
    generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
    for var, levels in dataset.items():
        data.append(dataset[var].expand_dims({'level': generic_level}, 1))

    data = xr.concat(data, 'level').transpose('time', 'level', 'lat', 'lon')
    mean = data.mean(('time', 'lat', 'lon')).compute()
    std = data.std('time').mean(('lat', 'lon')).compute()

    data = (data - mean) / std

    time = data.time
    data_ = data.sel(time=time)

    return data_, mean, std

def dataset_generator(dataset,history_size,target_size, rolling_step=1, sampling_step=1, single_step=False):
    data = []
    labels = []
    start_index = history_size
    end_index = len(dataset) - target_size + 1
    for i in range(start_index, end_index, rolling_step):
        indices = range(i-history_size, i, sampling_step)
        data.append(dataset[indices])
        indices1=range(i, i+target_size, sampling_step)
        if single_step:
            labels.append(dataset[i+target_size])
        else:
            labels.append(dataset[indices1])
    return np.array(data), np.array(labels)

class dataset_package(Dataset):
    def __init__(self, train_x, train_y):
        super().__init__()
        self.input = train_x
        self.target = train_y

    def GetDataShape(self):
        return {'input': self.input.shape,
                'target': self.target.shape}

    def __len__(self, ):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

if __name__ == '__main__':
    print('Configs:\n', configs.__dict__)

    data = np.load('/home/ubuntu/Project(Yu)/ocean/data/data_ocean.npz')
    train_x, train_y, test_x, test_y, mean, std = data['train_x'], data['train_y'], data['test_x'], data['test_y'], \
        data['mean'], data['std']
    dataset_train = dataset_package(train_x=train_x, train_y=train_y)
    dataset_test = dataset_package(train_x=test_x, train_y=test_y)
    del train_x, train_y, test_x, test_y
    print('Dataset_train Shape:\n', dataset_train.GetDataShape())
    print('Dataset_test Shape:\n', dataset_test.GetDataShape())

    # trainer = Trainer(configs)
    # trainer.save_configs('config.pkl')
    # trainer.train(dataset_train, dataset_test, 'checkpoint')

    #########################################################################################################################################

    device = configs.device
    model = SimpleCNN(configs).to(device)
    net = torch.load('checkpoint_134.15586853027344.chk')
    model.load_state_dict(net['net'])
    model.eval()

    data = DataLoader(dataset_test, batch_size=3, shuffle=False)

    with torch.no_grad():
        starttime = datetime.datetime.now()
        for i, (input, target) in enumerate(data):
            pred_temp = model(input.float().to(device))
            if i == 0:
                pred = pred_temp
                label = target
            else:
                pred = torch.cat((pred, pred_temp), 0)
                label = torch.cat((label, target), 0)
        endtime=datetime.datetime.now()
        print('SPEND TIME:',(endtime-starttime))

    np.savez('result134.15586853027344.npz', pred=pred.cpu(), label=label.cpu())
    ##########################################################################################################################################

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # device = configs.device
    # model = SpaceTimeTransformer(configs).to(device)
    # tensor = (torch.rand(1, 24, 4, 40, 40).to(device),torch.rand(1, 24, 4, 40, 40).to(device))
    #
    # with torch.no_grad():
    #     # 分析FLOPs
    #     flops = FlopCountAnalysis(model, tensor)
    #     print("FLOPs: ", flops.total())
    #
    #     # 分析parameters
    #     print(parameter_count_table(model))