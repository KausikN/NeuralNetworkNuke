# Normal FGSM
# Imports
from torch.autograd import Variable
# Imports

class FGSM:
    # Params
    
    # Params

    def __init__(self):
        self.enabled = True

    # Normal & Targeted FGSM
    def fgsm(self, x, y, net, loss_func, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        self.net = net
        self.loss_func = loss_func

        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv).reshape(1, -1)

        if targeted:
            cost = self.loss_func(h_adv, y)
        else:
            cost = -self.loss_func(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        h = self.net(x)
        h_adv = self.net(x_adv)

        return np.array(x_adv.detach()), np.array(h_adv.detach()), np.array(h.detach())
    # Normal & Targeted FGSM

    # Iterative FGSM
    def fgsm_it(self, x, y, net, loss_func, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        self.net = net
        self.loss_func = loss_func
        
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.loss_func(h_adv, y)
            else:
                cost = -self.loss_func(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = choose(x_adv > x+eps, x+eps, x_adv)
            x_adv = choose(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h
    # Iterative FGSM

    # Utils
    def choose(condition, a, b):
            condition = condition.float()
            return (condition*a) + ((1-condition)*b)
    # Utils