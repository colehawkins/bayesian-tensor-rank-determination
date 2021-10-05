from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import train,test,get_net
import time


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--model-type',
        default='full',
        choices=['CP', 'TensorTrain', 'TensorTrainMatrix','Tucker','full'],
    type=str)
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--rank-loss', type=bool, default=False)
    parser.add_argument('--kl-multiplier', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
    parser.add_argument('--em-stepsize', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
    parser.add_argument('--no-kl-epochs', type=int, default=5)
    parser.add_argument('--warmup-epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--prior-type', type=str, default='log_uniform')
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--tensorized', type=bool, default=False,
                        help='Run the tensorized model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = get_net(args).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        print("Epoch train time {:.2f}".format(time.time()-t))
        test(model, device, test_loader)


        if args.model_type == 'full':
            pass
        else:
            print("******Tensor Ranks*******")
            print(model.fc1.tensor.estimate_rank())
            print(model.fc2.tensor.estimate_rank())
            print("******Param Savings*******")
            param_savings_1 = model.fc1.tensor.get_parameter_savings(threshold=1e-4)
            param_savings_2 = model.fc2.tensor.get_parameter_savings(threshold=1e-4)
            full_params = 784*512+512+512*10+10
            print(param_savings_1,param_savings_2)
            total_savings = sum(param_savings_1)+sum(param_savings_2)
            print("Savings {} ratio {}".format(total_savings,full_params/(full_params-total_savings)))

            print("******End epoch stats*******")


if __name__ == '__main__':
    main()
