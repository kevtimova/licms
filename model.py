import torch.nn as nn

class Expert(nn.Module):

    def __init__(self, args):
        super(Expert, self).__init__()
        self.args = args

        # Architecture
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        if self.args.dataset == 'MNIST':
            self.model = nn.Sequential(
                *block(self.args.input_size, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(self.args.input_size)),
                nn.Tanh()
            )
        elif self.args.dataset == 'patient':
            self.model = nn.Sequential(
                *block(self.args.input_size, 128, normalize=False),
                *block(128, int(self.args.input_size))
            )
        else:
            raise NotImplementedError

    def forward(self, input):
        output = self.model(input)
        return output


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        # Architecture
        if self.args.dataset == 'MNIST':
            self.model = nn.Sequential(
                nn.Linear(self.args.input_size, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        elif self.args.dataset == 'patient':
            self.model = nn.Sequential(
                nn.Linear(self.args.input_size, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def forward(self, input):
        validity = self.model(input)
        return validity