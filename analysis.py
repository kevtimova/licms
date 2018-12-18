import argparse
import importlib
import json
import os
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets as datasets_torch
from torchvision import transforms

from model import Expert, Discriminator
from utils import init_weights

def test_trained_models(experts, discriminator, data_train, args, files):
    # Iterate through data
    for idx, batch in enumerate(data_train):
        _, x_transf = batch
        batch_size = x_transf.size(0)
        x_transf = x_transf.view(batch_size, -1).to(args.device)

        # Pass transformed data through experts
        exp_outputs = []
        expert_scores = []
        for i, expert in enumerate(experts):
            expert.eval()
            exp_output = expert(x_transf)
            exp_outputs.append(exp_output.view(batch_size, 1, args.input_size))
            exp_scores = discriminator(exp_output)
            expert_scores.append(exp_scores)

        # Get samples won by which expert
        exp_outputs = torch.cat(exp_outputs, dim=1)
        expert_scores = torch.cat(expert_scores, dim=1)
        mask_winners = expert_scores.argmax(dim=1)

        for i, expert in enumerate(experts):
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            n_expert_samples = winning_indexes.size(0)
            if n_expert_samples > 0:
                exp_samples = exp_outputs[winning_indexes, i].data
                np.savetxt(files[i], exp_samples.numpy(), delimiter=',')

    for i in range(args.num_experts):
        files[i].close()

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Learning Independent Causal Mechanisms')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--outdir', default='.', type=str,
                        help='path to the output directory')
    parser.add_argument('--dataset', default='patient', type=str,
                        help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--input_size', type=int, default=43, metavar='N',
                        help='input size of data (default: 784)')
    parser.add_argument('--num_experts', type=int, default=5, metavar='N',
                        help='number of experts (default: 5)')
    parser.add_argument('--model_for_initialization', type=str, default='',
                        help='path to pre-trained experts')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

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

    # Model
    experts = [Expert(args).to(args.device) for i in range(args.num_experts)]
    discriminator = Discriminator(args).to(args.device)

    # Load pre-trained experts and discriminator
    checkpt_dir = os.path.join(args.outdir, 'checkpoints')
    for i, expert in enumerate(experts):
        path_E = os.path.join(checkpt_dir,
                              args.model_for_initialization + '_E_{}.pth'.format(i + 1))
        init_weights(expert, path_E)

    path_D = os.path.join(checkpt_dir,
                          args.model_for_initialization + '_D.pth'.format(i + 1))
    init_weights(discriminator, path_D)

    # Load dataset
    if args.dataset in dir(datasets_torch):
        # Pytorch dataset
        dataset = getattr(datasets_torch, args.dataset)
        train_transform = transforms.Compose([transforms.ToTensor()])
        kwargs_train = {'download': True, 'transform': train_transform}
        dataset_train = dataset(root='{}/{}'.format(args.datadir, args.dataset), train=True, **kwargs_train)
    else:
        # Custom dataset
        dataset_train = getattr(importlib.import_module('{}'.format(args.dataset)), 'PatientsDataset')(args)

    # Create Dataloader from dataset
    data_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.cuda), pin_memory=args.cuda
    )

    # Files to store experts' output
    files = []
    analysis_dir = os.path.join(args.outdir, 'analysis')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    for i in range(args.num_experts):
        file_path = os.path.join(analysis_dir,
                                 '{}_E_{}_output.csv'.format(args.model_for_initialization, i + 1))
        f = open(file_path, 'ab')
        files.append(f)

    # Obtain outputs from experts
    test_trained_models(experts, discriminator, data_train, args, files)