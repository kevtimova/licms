import argparse
import json
import time
import importlib
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets as datasets_torch
from torchvision import transforms

from model import Expert, Discriminator

from tensorboardX import SummaryWriter

def initialize_expert(epochs, expert, i, optimizer, loss, data_train, args, writer):
    print("Initializing expert [{}] as identity on preturbed data".format(i+1))
    expert.train()

    for epoch in range(epochs):
        total_loss = 0
        n_samples = 0
        for batch in data_train:
            x_canonical, x_pret = batch
            batch_size = x_canonical.size(0)
            n_samples += batch_size
            x_pret = x_pret.view(x_pret.size(0), -1).to(args.device)
            x_hat = expert(x_pret)
            loss_rec = loss(x_pret, x_hat)
            total_loss += loss_rec.item()*batch_size
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

        # TODO: make sure loss is computed correctly
        mean_loss = total_loss/n_samples
        print("initialization epoch [{}] expert [{}] loss {:.4f}".format(epoch+1, i+1, mean_loss))
        writer.add_scalar('expert_{}_initialization_loss'.format(i+1), mean_loss, epoch+1)

def train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, args, writer):
    canonical_label = 1
    generated_label = 0
    total_loss_D_canon = 0
    total_loss_D_generated = 0
    n_samples = 0
    total_loss_expert = [0 for i in range(len(experts))]
    total_samples_expert = [0 for i in range(len(experts))]
    for batch in data_train:
        x_canon, x_pret = batch
        # x_pret = torch.randn(x_canon.size()) # TODO temporary since do not have the preturbed data yet
        batch_size = x_canon.size(0)
        n_samples += batch_size
        x_canon = x_canon.view(batch_size, -1).to(args.device)
        x_pret = x_pret.view(batch_size, -1).to(args.device)

        # Train Discriminator on canonical distribution
        scores = discriminator(x_canon)
        labels = torch.full((batch_size,), canonical_label, device=args.device).unsqueeze(dim=1)
        loss_D_canon = criterion(scores, labels)
        total_loss_D_canon += loss_D_canon.item()*batch_size
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
        total_loss_D_generated += loss_D_generated.item()*batch_size
        loss_D_generated.backward()
        optimizer_D.step()

        expert_scores = torch.cat(expert_scores, dim=1)
        mask_winners = expert_scores.argmax(dim=1)
        # Update each expert on samples it won
        for i, expert in enumerate(experts):
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            n_expert_samples = winning_indexes.size(0)
            total_samples_expert[i] += n_expert_samples
            if n_expert_samples > 0:
                samples = x_pret[winning_indexes]
                labels = torch.full((n_expert_samples,), canonical_label, device=args.device).unsqueeze(dim=1)
                optimizers_E[i].zero_grad()
                loss_E = criterion(discriminator(samples), labels)
                total_loss_expert[i] += loss_E.item()*n_expert_samples
                loss_E.backward()
                optimizers_E[i].step()

    # Logging
    mean_loss_D_generated = total_loss_D_generated/n_samples
    mean_loss_D_canon = total_loss_D_canon/n_samples
    print("epoch [{}] loss_D_generated {:.4f}".format(epoch+1, mean_loss_D_generated))
    print("epoch [{}] loss_D_canon {:.4f}".format(epoch+1, mean_loss_D_canon))
    writer.add_scalar('loss_D_canonical', mean_loss_D_canon, epoch+1)
    writer.add_scalar('loss_D_generated', mean_loss_D_generated, epoch+1)
    for i in range(len(experts)):
        if total_samples_expert[i]> 0:
            mean_loss_expert = total_loss_expert[i]/total_samples_expert[i]
            print("epoch [{}] expert [{}] loss {:.4f}".format(epoch+1, i+1, mean_loss_expert))
            writer.add_scalar('expert_{}_loss'.format(i+1), mean_loss_expert, epoch+1)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Learning Independent Causal Mechanisms')
    parser.add_argument('--datadir', default='./data', type=str,
                        help='path to the directory that contains the data')
    parser.add_argument('--outdir', default='.', type=str,
                        help='path to the output directory')
    parser.add_argument('--dataset', default='patient_data', type=str,
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
    parser.add_argument('--learning_rate_expert', type=float, default=1e-3,
                        help='size of expert learning rate')
    parser.add_argument('--learning_rate_discriminator', type=float, default=1e-3,
                        help='size of discriminator learning rate')
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
        name = '{}_n_exp_{}_bs_{}_lre_{}_lrd_{}_ei_{}_e_{}_{}'.format(
            args.dataset, args.num_experts, args.batch_size, args.learning_rate_expert,
            args.learning_rate_discriminator, args.epochs_init, args.epochs, timestamp)
        args.name = name
    else:
        args.name = '{}_{}'.format(args.name, timestamp)
    print('\nExperiment: {}\n'.format(args.name))

    # Logging. To run: tensorboard --logdir <args.outdir>/logs
    log_dir = os.path.join(args.outdir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir_exp = os.path.join(log_dir, args.name)
    os.mkdir(log_dir_exp)
    writer = SummaryWriter(log_dir=log_dir_exp)

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

    # Model
    experts = [Expert(args).to(args.device) for i in range(args.num_experts)]
    discriminator = Discriminator(args).to(args.device)

    # Losses
    loss_initial = torch.nn.MSELoss() # TODO: mean or sum reduction
    criterion = torch.nn.BCELoss() # TODO: pick the right loss

    # Optimizers
    optimizers_E = []
    for i in range(args.num_experts):
        optimizer_E = torch.optim.Adam(experts[i].parameters(), lr=args.learning_rate_expert, weight_decay=args.weight_decay)
        optimizers_E.append(optimizer_E)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator, weight_decay=args.weight_decay)

    # Initialize Experts as approximately Identity
    for i, expert in enumerate(experts):
        initialize_expert(args.epochs_init, expert, i, optimizers_E[i], loss_initial, data_train, args, writer)

    # Training
    for epoch in range(args.epochs):
        train_system(epoch, experts, discriminator, optimizers_E, optimizer_D, criterion, data_train, args, writer)


