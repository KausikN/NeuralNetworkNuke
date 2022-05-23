"""
FGSM Neural Network Attack
"""

# Imports
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Main Classes
class FGSM:
    '''
    FGSM Attack Module
    '''

    # Init
    def __init__(self):
        '''
        Init
        '''
        self.enabled = True

    # Normal & Targeted FGSM
    def fgsm(self, x, y, net, loss_func, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        '''
        Normal & Targeted FGSM
        '''
        self.net = net
        self.loss_func = loss_func

        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv).reshape(1, -1)
        # print("DEBUG: h_adv shape: ", h_adv.shape, " - y shape: ", y.shape, " <-> y: ", y)

        cost = self.loss_func(h_adv, y)
        # Cost should be negative if not targeted
        if not targeted: cost = -cost 

        self.net.zero_grad()
        if x_adv.grad is not None: x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return np.array(x_adv.detach()), np.array(h_adv.detach()), np.array(h.detach())

# Display Classes
def FGSM_Display_MNIST(model, X, Y, y_target=None, eps=0.25, cmap="Greys", verbose=False):
    '''
    Display FGSM Attack
    '''
    # Targeted Changes
    targeted = (y_target is not None)
    if targeted: y_target = torch.from_numpy(np.array(y_target).reshape(-1))
    # Init
    x = np.array(X)
    x_flattened = torch.from_numpy(x.reshape(1, -1))
    y_pred = y_target if targeted else torch.from_numpy(np.array(Y).reshape(-1))
    # Apply FGSM
    att_fgsm = FGSM()
    # x_adv_flattened - new image
    # h_adv - predicted for x_adv_flattened
    # h - predicted for original x
    x_adv_flattened, h_adv, h = att_fgsm.fgsm(x_flattened, y_pred, model.net, model.loss_fn, True, eps)
    # Get Outputs
    x_flattened = np.array(x_flattened)
    x_adv = x_adv_flattened.reshape(X.shape)
    added_noise_flattened = x_adv_flattened - x_flattened
    added_noise = added_noise_flattened.reshape(X.shape)

    # Display
    if verbose:
        print("X:", x.shape)
        print("x_flattened:", x_flattened.shape)
        print("x_adv:", x_adv.shape)
        print("x_adv_flattened:", x_adv_flattened.shape)
        print("Y_pred before attack:", str(np.argmax(h)))
        print("Y_pred after attack:", str(np.argmax(h_adv)))
        print("Y_pred Values before attack:", str(h))
        print("Y_pred Values after attack:", str(h_adv))
    # Plot
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(1, 3, 1)
    plt.imshow(x[0], cmap=cmap)
    fig.add_subplot(1, 3, 2)
    plt.imshow(added_noise, cmap=cmap)
    fig.add_subplot(1, 3, 3)
    plt.imshow(x_adv, cmap=cmap)
    # plt.show()

    OUTPUTS = {
        "plot": fig
    }
    return OUTPUTS

# Main Vars
ATTACK_FGSM = {
    "mnist": {
        "class": FGSM,
        "display": FGSM_Display_MNIST
    }
}