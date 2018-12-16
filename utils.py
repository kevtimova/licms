import torch

class LossReconstruction:

    def __init__(self, args):
        self.args = args

    def MSE(self, target, pred):
        sq_diff = torch.pow(target - pred, 2)
        sum_sq_diff = sq_diff.view(sq_diff.size(0), -1).sum(1)
        return sum_sq_diff.mean()


def init_weights(expert, path):
    pre_trained_dict = torch.load(path, map_location=lambda storage, loc: storage)
    for layer in pre_trained_dict.keys():
        expert.state_dict()[layer].copy_(pre_trained_dict[layer])
    for param in expert.parameters():
        param.requires_grad = True