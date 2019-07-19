import torch

class SoftMax_NLLL(torch.nn.Module):
    def __init__(self, dim=1):
        self.dim = dim
        self.sm = torch.nn.Softmax2d()
        # NLLLoss expects inputs to be (minibatch, C, d1, d2)
        self.nlll = torch.nn.N