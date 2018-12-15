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

        self.model_mnist = nn.Sequential(
            *block(self.args.input_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(self.args.input_size)),
            nn.Tanh()
        )

        self.model_patient = nn.Sequential(
            *block(self.args.input_size, 64, normalize=False),
            *block(64, int(self.args.input_size))
        )

    def forward(self, input):
        if self.args.dataset == 'MNIST':
            output = self.model_mnist(input)
        elif self.args.dataset == 'patient_data':
            output = self.model_patient(input)
        else:
            raise NotImplementedError
        return output


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        # Architecture
        self.model_mnist = nn.Sequential(
            nn.Linear(self.args.input_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.model_patient = nn.Sequential(
            nn.Linear(self.args.input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if self.args.dataset == 'MNIST':
            validity = self.model_mnist(input)
        elif self.args.dataset == 'patient_data':
            validity = self.model_patient(input)
        return validity