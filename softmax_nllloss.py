import torch

class SoftMax_NLLL(torch.nn.Module):
    # This class implements cross entropy. Although there is already a cross entropy in PyTorch, we had to do the cross entropy in dim=1
    def __init__(self, dim=1):
        super(SoftMax_NLLL, self).__init__()
        self.dim = dim
        self.sm = torch.nn.LogSoftmax(self.dim)
        # NLLLoss expects inputs to be (minibatch, C, d1, d2)
        self.nlll = torch.nn.NLLLoss()
    def forward(self, x, target):
        lsm = self.sm(x)
        loss = self.nlll(lsm, target)
        return loss