import torch

class LossReconstruction:

    def __init__(self, args):
        self.args = args

    def MSE(self, target, pred):
        sq_diff = torch.pow(target - pred, 2)
        sum_sq_diff = sq_diff.view(sq_diff.size(0), -1).sum(1)
        return sum_sq_diff.mean()