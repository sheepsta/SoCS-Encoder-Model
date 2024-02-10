import torch
import torch.utils.data as data
import numpy as np
import os
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
        self.encoder = nn.Sequential(
            nn.Linear(2800, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 28)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(28, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 2800)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


mse_loss = nn.MSELoss()


def contractive_loss_function(W, x, recons_x, h, lam=1e-3):
    mse = mse_loss(recons_x, x)
    contractive_loss = torch.sum(torch.pow(W, 2))
    return mse + lam * contractive_loss


def main():
    # Adjust these paths and parameters as needed
    dataset_path = '/path/to/your/dataset'
    model_path = './model_cae'
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load your dataset
    obs = load_dataset()  # Make sure this function is defined and loads your dataset correctly

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(0, len(obs), batch_size):
            encoder.zero_grad()
            decoder.zero_grad()

            batch_obs = obs[i:i+batch_size]
            batch_obs = torch.from_numpy(batch_obs).float().to(device)

            encoded_repr = encoder(batch_obs)
            decoded_obs = decoder(encoded_repr)

            # Fetch the weights of the last layer of the encoder
            W = list(encoder.parameters())[-2]
            loss = contractive_loss_function(
                W, batch_obs, decoded_obs, encoded_repr, lam=1e-3)

            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the models
    torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder.pth'))


if __name__ == '__main__':
    main()
