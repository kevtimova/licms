import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import json
import numpy as np
import time

from model import Expert, Discriminator

def initialize_expert(epochs, expert, optimizer, loss, data_train, args):
    expert.train()
    losses = []

    for epoch in range(epochs):
        for batch in data_train:
            x_real, x_pret = batch
            x_real = Variable(x_real).to(args.device)
            x_hat = expert(x_real)
            loss_rec = loss(x_real, x_hat)
            losses.append(loss_rec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: make sure loss is computed correctly
        mean_loss = np.array([loss.cpu().data.item() for loss in losses]).mean()
        print("epoch [{}] {}".format(epoch+1, mean_loss))

def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, loss, data_train, args):
    canonical_label = 1
    generated_label = 0
    losses = []
    for batch in data_train:
        batch_size = None # TODO
        x_canon, x_pret = batch
        x_canon = Variable(x_canon).to(args.device)
        x_pret = Variable(x_pret).to(args.device)

        # Train Discriminator on canonical distribution
        scores = discriminator(x_canon)
        labels = torch.full((batch_size,), canonical_label, device=args.device)
        loss_D_cannon = loss(scores, labels)
        optimizer_D.zero_grad()
        loss_D.backward()

        # Train Discriminator on experts output
        outputs = []
        labels.fill_(generated_label)
        loss_D_generated = 0 # TODO make compatible with output
        expert_scores = torch.FloatTensor().cuda() if args.cuda else torch.FloatTensor()
        for i, expert in enumerate(experts):
            output = expert(x_pret)
            scores = discriminator(output.detach())
            torch.cat(expert_scores, scores) # TODO check dim
            loss_D_generated += criterion(output, scores)
        loss_D_generated = loss_D_generated / args.num_experts
        loss_D_generated.backward()
        optimizer_D.step()

        expert_winners = None # TODO best expert for each data point
        # TODO update each expert
        for i, expert in enumerate(experts):
            labels.fill_(canonical_label)
            optimizers_E[i].zero_grad()




if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Learning Independent Causal Mechanisms')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--dataset', default='MIMIC', type=str,
                        help='name of the dataset')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='optimization algorithm (options: sgd | adam, default: adam)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
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
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')

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

    # Load Data
    data_train = None # TODO: load data

    # Model
    experts = [Expert(args) for i in range(args.num_experts)]
    discriminator = Discriminator(args)

    # Losses
    loss_initial = torch.nn.MSELoss() # TODO: mean or sum reduction
    criterion = torch.nn.BCELoss() # TODO: pick the right loss

    # Optimizers
    optimizers_E = []
    for i in range(args.num_experts)
        optimizer_E = torch.optim.Adam(experts[i].parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizers_E.append(optimizer_E)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize Experts as approximately Identity
    for i, expert in enumerate(experts):
        initialize_expert(args.epochs_init, expert, optimizers_E[i], loss_init, data_train, args)

    # Training
    for epoch in range(args.epochs):

        train_system()


