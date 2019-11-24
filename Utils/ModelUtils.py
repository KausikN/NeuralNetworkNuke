# Model Utils
def fix_act_funcs_size(size, act_funcs):
    afsize = len(act_funcs)
    fixed_act_funcs = []
    for i in range(size):
        if i < afsize:
            fixed_act_funcs.append(act_funcs[i])
        else:
            fixed_act_funcs.append(act_funcs[afsize-1])

    return fixed_act_funcs


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()

# One Hot Encoding
def OneHotEncode(X):
    Y = []
    for x in X:
        ohe = np.zeros(10)
        ohe[x] = 1
        Y.append(ohe)
        # print(x)
        # print(ohe)
    Y = np.array(Y)
    print(torch.from_numpy(Y.astype(int)).shape)
    return torch.from_numpy(Y.astype(int))