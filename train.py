import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.autograd import grad

# from generator import Generator3, Generator64
# from discriminator import Discriminator64

from dcgan import Generator, Discriminator, weights_init

from utils import weights_init
from losses import loss_dcgan_dis, loss_dcgan_gen

# import wandb
# from wandb.pytorch import Wand

# wandb.init(project='DLHW3-CelebA-GAN')

class R1(torch.nn.Module):
    """
    Implementation of the R1 GAN regularization.
    """

    def __init__(self):
        """
        Constructor method
        """
        # Call super constructor
        super(R1, self).__init__()
        self.r1_w = 0.2

    def forward(self, prediction_real: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the regularization
        :param prediction_real: (torch.Tensor) Prediction of the discriminator for a batch of real images
        :param real_sample: (torch.Tensor) Batch of the corresponding real images
        :return: (torch.Tensor) Loss value
        """
        # Calc gradient
        grad_real = torch.autograd.grad(outputs=prediction_real.sum(), inputs=real_sample, create_graph=True)[0]
        # Calc regularization
        regularization_loss: torch.Tensor = self.r1_w * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return regularization_loss

def r1loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Loss functions
criterion_cycle = torch.nn.L1Loss()

# Loss weights
lambda_gp = 10

REAL_LABEL = 1
FAKE_LABEL = 0

z_discrete = 20
z_continuous = 80
latent_dim = z_discrete + z_continuous


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lrg", type=float, default=0.00001, help="adam: learning rate of the generator")
    parser.add_argument("--lrd", type=float, default=0.00001, help="adam: learning rate of the discriminator")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="adam: L2 regularization coefficient")
    parser.add_argument("--gp", type=int, default=10, help="Gradient penalty parameter")
    parser.add_argument("--ngf", type=int, default=64, help="Number of generator features")
    parser.add_argument("--ndf", type=int, default=64, help="Number of discriminator features")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
    parser.add_argument("--print_interval", type=int, default=10, help="print every X batches")
    parser.add_argument("--generator_path", type=str, default=None, help="path to pre-trained generator weights")
    parser.add_argument("--discriminator_path", type=str, default=None, help="path to pre-trained generator weights")
    parser.add_argument("--discriminator_hold", type=int, default=1, help="train the discriminator every x batches")
    args = parser.parse_args()
    print(args)

    cuda = True if torch.cuda.is_available() else False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_dataset = dsets.CelebA(
        root='./',
        split = 'train',
        transform=transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.RandomCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        ),
        download=False
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True)

    x,y = train_dataset[0]
    img_shape = tuple(x.shape)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator
    generator = Generator(latent_dim, args.ngf).to(device)
    generator.apply(weights_init)
    if args.generator_path:
        generator.load_state_dict(torch.load(args.generator_path, map_location=device))

    # Initialize discriminator
    discriminator = Discriminator(args.channels, args.ndf).to(device)
    discriminator.apply(weights_init)
    if args.discriminator_path:
        discriminator.load_state_dict(torch.load(args.discriminator_path, map_location=device))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lrg, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lrd, betas=(args.b1, args.b2), weight_decay=args.weight_decay)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    dt = datetime.now().strftime('%y_%m_%d_%H_%M')

    os.makedirs("debug", exist_ok=True)
    os.makedirs(f"debug/{dt}", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ----------
    #  Training
    # ----------

    results = []
    # Initialize results file
    with open(f'results/{dt}.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['d_loss', 'g_loss', 'epoch', 'batch'])
        writer.writeheader()

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(train_loader):


            # Transfer data tensor to GPU/CPU (device)
            real_data = imgs.to(device)
            
            # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
            b_size = real_data.size(0)
            
            # Make accumalated gradients of the discriminator zero.
            discriminator.zero_grad()
            # Create labels for the real data. (label=1)
            label = torch.full((b_size, ), REAL_LABEL, device=device).float()
            output = discriminator(real_data).view(-1)
            d_loss_real = adversarial_loss(output, label)

            d_loss_real.backward()
            
            # Random int from [0,10]
            zd = torch.Tensor(np.random.randint(0, 10, (imgs.shape[0], z_discrete, 1, 1)))
            
            # Uniform random continuous from (-1,1)
            zc = 2*torch.Tensor(np.random.rand(imgs.shape[0], z_continuous, 1, 1)) - 1
            noise = torch.cat([zd, zc], dim=1).to(device)

            gen_imgs = generator(noise)
            
            label.fill_(FAKE_LABEL  )
            
            output = discriminator(gen_imgs.detach()).view(-1)
            d_loss_fake = adversarial_loss(output, label)
            
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()
            
            # Make accumalted gradients of the generator zero.
            generator.zero_grad()

            label.fill_(REAL_LABEL)

            output = discriminator(gen_imgs).view(-1)
            g_loss = adversarial_loss(output, label)

            g_loss.backward()

            D_G_z2 = output.mean().item()
            
            # Update generator parameters.
            optimizer_G.step()


            # # Adversarial ground truths
            # valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # # Configure input
            # real_imgs = Variable(imgs.type(Tensor))

            # # -----------------
            # #  Train Generator
            # # -----------------

            # optimizer_G.zero_grad()

            # zd = torch.Tensor(np.random.randint(0, 10, (imgs.shape[0], z_discrete)))
            # zc = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], z_continuous)))
            # z = torch.cat([zd, zc], dim=1).to(device)

            # # Generate a batch of images
            # gen_imgs = generator(z)

            # # Loss measures generator's ability to fool the discriminator
            # g_loss = adversarial_loss(discriminator(gen_imgs).view(valid.shape), valid)
            # # g_loss = loss_dcgan_gen(discriminator(gen_imgs).view(valid.shape))

            # g_loss.backward()
            # optimizer_G.step()

            # # ---------------------
            # #  Train Discriminator
            # # ---------------------

            # if i % args.discriminator_hold == 0:
            #     optimizer_D.zero_grad()

            #     # zd = torch.Tensor(np.random.randint(0, 10, (imgs.shape[0], z_discrete)))
            #     # zc = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], z_continuous)))
            #     # z = torch.cat([zd, zc], dim=1).to(device)

            #     # Generate a batch of images
            #     # gen_imgs = generator(z)

            #     # Measure discriminator's ability to classify real from generated samples
            #     prediction_real = discriminator(real_imgs).view(valid.shape)
            #     prediction_fake = discriminator(gen_imgs.detach()).view(fake.shape)
            #     # real_loss = adversarial_loss(prediction_real, valid)
            #     # fake_loss = adversarial_loss(prediction_fake, fake)

            #     # imgs.requires_grad = True

            #     # gradient_penalty = lambda_gp * compute_gradient_penalty(discriminator, real_imgs.detach(), gen_imgs.detach())
            #     # d_loss = (real_loss + fake_loss) / 2 + gradient_penalty
            #     # d_loss = 0.5 * (real_loss + fake_loss)

            #     real_loss, fake_loss = loss_dcgan_dis(prediction_fake, prediction_real)
            #     d_loss = real_loss + fake_loss
            #     d_loss.backward(retain_graph=True)
            #     optimizer_D.step()

            batches_done = epoch * len(train_loader) + i
            if batches_done % args.sample_interval == 0:
                print(f'Saving image debug/{batches_done}.png')
                img_grid = make_grid(gen_imgs.data[:16])
                save_image(img_grid, f"debug/{dt}/{batches_done}.png")

            if i % args.print_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )

                batch_result = {
                    'd_loss': float(d_loss.item()),
                    'g_loss': float(g_loss.item()),
                    'epoch': epoch,
                    'batch': batches_done
                }

                with open(f'results/{dt}.csv', 'a+') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(batch_result.keys()))
                    writer.writerow(batch_result)

        print('Saving models')
        torch.save(generator.state_dict(), f'models/generator_{dt}.model')
        torch.save(discriminator.state_dict(), f'models/discriminator_{dt}.model')
