import argparse
import json
import time
import numpy as np
import importlib

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets as datasets_torch
from torchvision import transforms

from model import Expert, Discriminator


def initialize_expert(epochs, expert, i, optimizer, loss, data_train, args):
    print("Initializing expert {} as identity on preturbed data".format(i))
    expert.train()
    losses = []

    for epoch in range(epochs):
        for batch in data_train:
            x_canonical, x_pret = batch
            x_canonical = x_canonical.view(x_canonical.size(0), -1)
            x_canonical = Variable(x_canonical).to(args.device)
            x_hat = expert(x_canonical)
            loss_rec = loss(x_canonical, x_hat)
            losses.append(loss_rec)
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

        # TODO: make sure loss is computed correctly
        mean_loss = np.array([loss.cpu().data.item() for loss in losses]).mean()
        print("epoch [{}] {}".format(epoch+1, mean_loss))

def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, loss, data_train, args):
    canonical_label = 1
    generated_label = 0
    losses = []
    for batch in data_train:
        x_canon, x_pret = batch
        x_pret = torch.randn(x_canon.size()) # TODO temporary since do not have the preturbed data yet
        batch_size = x_canon.size(0)
        x_canon = x_canon.view(batch_size, -1)
        x_pret = x_pret.view(batch_size, -1)
        x_canon = Variable(x_canon).to(args.device)
        x_pret = Variable(x_pret).to(args.device)

        # Train Discriminator on canonical distribution
        scores = discriminator(x_canon)
        labels = torch.full((batch_size,), canonical_label, device=args.device).unsqueeze(dim=1)
        loss_D_canon = loss(scores, labels)
        optimizer_D.zero_grad()
        loss_D_canon.backward()

        # Train Discriminator on experts output
        labels.fill_(generated_label)
        loss_D_generated = 0 # TODO make compatible with output
        expert_scores = []
        for i, expert in enumerate(experts):
            output = expert(x_pret)
            scores = discriminator(output.detach())
            expert_scores.append(scores)
            loss_D_generated += criterion(scores, labels)
        loss_D_generated = loss_D_generated / args.num_experts
        loss_D_generated.backward()
        optimizer_D.step()

        expert_scores = torch.cat(expert_scores, dim=1)
        mask_winners = expert_scores.argmax(dim=1)
        # Update each expert on samples it won
        for i, expert in enumerate(experts):
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            n_samples = winning_indexes.size(0)
            if n_samples > 0:
                samples = x_pret[winning_indexes]
                labels = torch.full((n_samples,), canonical_label, device=args.device).unsqueeze(dim=1)
                optimizers_E[i].zero_grad()
                loss_E = criterion(discriminator(samples), labels)
                loss_E.backward()
                optimizers_E[i].step()

    # TODO print losses
    print("epoch [{}] losses: ".format(epoch))

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Learning Independent Causal Mechanisms')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--dataset', default='MIMIC', type=str,
                        help='name of the dataset')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--input_size', type=int, default=784, metavar='N',
                        help='input size of data (default: 784)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs_init', type=int, default=100, metavar='N',
                        help='number of epochs to initially train experts (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='size of learning rate')
    parser.add_argument('--name', type=str, default='',
                        help='name of experiment')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for optimizer')
    parser.add_argument('--num_experts', type=int, default=5, metavar='N',
                        help='number of experts (default: 5)')

    # Get arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Random seed
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Experiment name
    timestamp = str(int(time.time()))
    if args.name == '':
        args.name = '{}_{}'.format(args.dataset, timestamp)
    else:
        args.name = '{}_{}'.format(args.name, timestamp)
    print('\nExperiment: {}\n'.format(args.name))

    # Load dataset
    if args.dataset in dir(datasets_torch):
        # Pytorch dataset
        dataset = getattr(datasets_torch, args.dataset)
        train_transform = transforms.Compose([transforms.ToTensor()])
        kwargs_train = {'download': True, 'transform': train_transform}
        dataset_train = dataset(root='{}/{}'.format(args.datadir, args.dataset), train=True, **kwargs_train)
    else:
        # Custom dataset
        train_transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = getattr(importlib.import_module('data.{}'.format(args.dataset)), 'PatientsDataset')(args.datadir, train_transform)

    # Create Dataloader from dataset
    data_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.cuda), pin_memory=args.cuda
    )

    # Model
    experts = [Expert(args) for i in range(args.num_experts)]
    discriminator = Discriminator(args)

    # Losses
    loss_initial = torch.nn.MSELoss() # TODO: mean or sum reduction
    criterion = torch.nn.BCELoss() # TODO: pick the right loss

    # Optimizers
    optimizers_E = []
    for i in range(args.num_experts):
        optimizer_E = torch.optim.Adam(experts[i].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizers_E.append(optimizer_E)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize Experts as approximately Identity
    for i, expert in enumerate(experts):
        initialize_expert(args.epochs_init, expert, i, optimizers_E[i], loss_initial, data_train, args)

    # Training
    for epoch in range(args.epochs):

        train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, args)


