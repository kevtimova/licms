import torch
import torch.nn as nn

import argparse
import json
import numpy as np
import time

from model import Expert, Discriminator

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

    # Model
    expert = Expert(args)
    discriminator = Discriminator(args)

    # Losses
    loss = torch.nn.BCELoss()

    # Optimizers
    optimizer_E = torch.optim.Adam(expert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Initialize Experts as approximately Identity

    # Training
    for epoch in range(args.epochs):
        # Train Discriminator

        # Train Experts

