import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Generator2(nn.Module):
    # initializers
    def __init__(self, latent_dim=100, d=128):
        super(Generator2, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(d*4, d*2, 2, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(d*2, d, 5, 1, 1),
            nn.BatchNorm2d(d),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    # forward method
    def forward(self, z):
        z = self.block1(z)
        z = self.block2(z)
        z = self.block3(z)
        z = self.block4(z)
        z = self.block5(z)
        z = self.block6(z)
        return z

class Generator3(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self, latent_dim=100):
        super(Generator3, self).__init__()

        # Project and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True))

        # Upsample
        self.features = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        return self.features(x)


class Generator4(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self, latent_dim=100):
        super(Generator4, self).__init__()

        # Project and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.Linear(512 * 4 * 4, 1024 * 4 * 4, bias=False),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(inplace=True))

        # Upsample
        self.features = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        return self.features(x)

class Generator5(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self, latent_dim=100):
        super(Generator5, self).__init__()

        # Project and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.Linear(512 * 4 * 4, 1024 * 4 * 4, bias=False),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(inplace=True))

        # Upsample
        self.features = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 3, 1, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        return self.features(x)


class Generator64(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self, latent_dim=100):
        super(Generator64, self).__init__()

        # Project and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 1024 * 4 * 4, bias=False),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.LeakyReLU(inplace=True))

        # Upsample
        self.features = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=7, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        return self.features(x)

if __name__ == '__main__':
    import torch 

    cuda = True if torch.cuda.is_available() else False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    latent_dim = 100

    # Initialize generator
    generator = Generator64(latent_dim).to(device)

    b = 5

    z = torch.Tensor(np.random.normal(0, 1, (b, latent_dim))).to(device)
    gen_imgs = generator(z)
    print('generated images shape:', gen_imgs.shape)
