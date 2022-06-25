import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator2(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator2, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1),
            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(),
            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(),
            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(d*8),
            nn.ReLU(),
            nn.Conv2d(d*8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, x):
        return self.disc(x)

class Discriminator3(nn.Module):
    def __init__(self):
        super(Discriminator3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1))

    def forward(self, x):
        return self.features(x).view(-1)


class Discriminator4(nn.Module):
    def __init__(self):
        super(Discriminator4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1)
        )

    def forward(self, x):
        return self.features(x).view(-1)


if __name__ == '__main__':
    import torch
    import numpy as np

    cuda = True if torch.cuda.is_available() else False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Initialize generator
    D = Discriminator4().to(device)
    z = torch.Tensor(np.random.normal(0, 1, (2, 3 , 64, 64))).to(device)
    output = D(z)
    print('output shape:', output.shape)