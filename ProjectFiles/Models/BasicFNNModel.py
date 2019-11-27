# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Utils import fix_act_funcs_size, kaiming_init
# Imports

class BasicFNNModel(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_sizes, act_funcs=[nn.Sigmoid()], out_act_funcs=[nn.Softmax(dim=1)]):
        super(BasicFNNModel, self).__init__()
        torch.manual_seed(0)

        # Fix act func sizes
        act_funcs = fix_act_funcs_size(len(hidden_sizes), act_funcs)

        seq = []

        seq.append(('input_layer', nn.Linear(n_inputs, hidden_sizes[0])))
        seq.append(('input_act_layer', act_funcs[0]))

        for i in range(1, len(hidden_sizes)):
            seq.append(('hidden_' + str(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i])))
            seq.append(('hidden_act_layer_' + str(i), act_funcs[i]))

        seq.append(('output_layer', nn.Linear(hidden_sizes[len(hidden_sizes)-1], n_outputs)))
        for index in range(len(out_act_funcs)):
            seq.append(('output_act_layer_' + str(index), out_act_funcs[index]))

        seqdict = OrderedDict(seq)

        print("Network Sequence: ", seqdict, "\n")

        self.net = nn.Sequential(
            seqdict
        )

    def getNetwork(self):
        return self.net

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        return self.net(X).detach().numpy()

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())

    def fit(self, x, y, opt, loss_fn, epochs = 1000, display_loss=True):
        from torch import optim
        import matplotlib.pyplot as plt
        import matplotlib.colors
        import numpy as np
        import copy

        self.loss_fn = loss_fn

        #for i in y_OH:
        y = y.nonzero()[:, 1]

        # x = x.unsqueeze(0)

        self.min_loss_config = None
        self.min_loss = -1

        loss_arr = []
        from tqdm import tqdm_notebook 
        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
            self.cur_iter = i
        # for epoch in range(epochs):
            # for outi in range(y.shape[1]):
            #     y_pred = self.forward(x)
            #     loss += loss_fn(y_pred[outi], y[outi])
            loss = loss_fn(self.forward(x), y)
            loss_temp = loss.item()
            loss_arr.append(loss_temp)

            loss.backward()
            opt.step()
            opt.zero_grad()

            if (self.min_loss == -1 or self.min_loss >= loss_temp):
                self.min_loss = loss_temp
                self.min_loss_config = copy.copy(self)

        if display_loss:
            plt.plot(loss_arr)
            plt.xlabel('Epochs')
            plt.ylabel('CE')
            plt.show()

        return loss_arr