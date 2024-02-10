import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
import argparse
import torchvision
from torch import nn
from torch.autograd import Variable


def load_dataset(N=30000, NP=1800):

    obstacles = np.zeros((N, 2800), dtype=np.float32)
    for i in range(0, N):
        print(i)
        temp = np.fromfile(
            '/Users/juliusarolovitch/SoCS-Encoder-Model/dataset/obs_cloud/obc' + str(i) + '.dat')
        temp = temp.reshape(len(temp)//2, 2)
        obstacles[i] = temp.flatten()

    return obstacles


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2800, 512), nn.PReLU(), nn.Linear(
            512, 256), nn.PReLU(), nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 28))

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(28, 128), nn.PReLU(), nn.Linear(
            128, 256), nn.PReLU(), nn.Linear(256, 512), nn.PReLU(), nn.Linear(512, 2800))

    def forward(self, x):
        x = self.decoder(x)
        return x


mse_loss = nn.MSELoss()
lam = 1e-3


def loss_function(W, x, recons_x, h):
    mse = mse_loss(recons_x, x)
    """
	W is shape of N_hidden x N. So, we do not need to transpose it as opposed to http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
	"""
    dh = h*(1-h) 
    contractive_loss = torch.sum(Variable(W)**2, dim=1).sum().mul_(lam)
    return mse + contractive_loss


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    obs = load_dataset()

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adagrad(params)
    total_loss = []
    for epoch in range(args.num_epochs):
        print("Epoch", epoch)
        avg_loss = 0
        for i in range(0, len(obs), args.batch_size):
            decoder.zero_grad()
            encoder.zero_grad()
            end_index = i + args.batch_size if i + \
                args.batch_size < len(obs) else len(obs)
            inp = obs[i:end_index]
            inp = torch.from_numpy(inp).to(device)
            # Forward pass
            h = encoder(inp)
            output = decoder(h)
            W = encoder.state_dict()['encoder.6.weight']
            loss = loss_function(W, inp, output, h)
            avg_loss += loss.item()
            # Backward and optimize
            loss.backward()
            optimizer.step()
        avg_loss /= (len(obs) / args.batch_size)
        print("--Average Loss:", avg_loss)
        total_loss.append(avg_loss)

    avg_loss = 0
    for i in range(len(obs)-5000, len(obs), args.batch_size):
        inp = obs[i:i+args.batch_size]
        inp = torch.from_numpy(inp).to(device) 
        # Forward pass
        output = encoder(inp)
        output = decoder(output)
        loss = mse_loss(output, inp)
        avg_loss += loss.item()  
    print("--Validation average loss:")
    print(avg_loss / (5000 / args.batch_size))

    torch.save(encoder.state_dict(), os.path.join(
        args.model_path, 'cae_encoder.pkl'))
    torch.save(decoder.state_dict(), os.path.join(
        args.model_path, 'cae_decoder.pkl'))
    torch.save(total_loss, 'total_loss.dat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/', help='path for saving trained models')
    parser.add_argument('--no_env', type=int, default=50,
                        help='directory for obstacle images')
    parser.add_argument('--no_motion_paths', type=int, default=2000,
                        help='number of optimal paths in each environment')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    parser.add_argument('--input_size', type=int, default=18,
                        help='dimension of the input vector')
    parser.add_argument('--output_size', type=int, default=2,
                        help='dimension of the input vector')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
