
import argparse
import os
import numpy as np
from time import time

import torch
from torch import autograd
# import torch.autograd as autograd
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_processing import fetch_dataset, save_csv
from gan import Generator, Discriminator


preset_img_shape = (3, 64, 64)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    k = 2
    p = 6

    opt = parse_argument()
    img_shape = preset_img_shape
    # img_shape = (opt.channels, opt.img_size, opt.img_size)

    # root_path = "C:/Users/Lucky/PycharmProjects/finalproject/"
    # data_path = root_path + "face/"
    # work_path = root_path + "workspace/"
    # save_path = root_path + "workspace/new_picture/"
    # os.makedirs(work_path, exist_ok=True)
    # os.makedirs(save_path, exist_ok=True)

    root_path = "/Users/yuming/OneDrive/sync/semester/ML/hw/project/dataset/"
    data_path = root_path + "faces/"
    save_path = root_path + "fake-faces/"
    save_csv_loss_g = root_path + "csv/loss_g.csv"
    save_csv_loss_d = root_path + "csv/loss_d.csv"

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, img_shape)
    discriminator = Discriminator(img_shape)

    cuda_enabled = torch.cuda.is_available()
    if cuda_enabled:
        generator.cuda()
        discriminator.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
    # Tensor = torch.cuda.FloatTensor if cuda_enabled else torch.FloatTensor

    image_raw_data = fetch_dataset(data_path)
    data_loader = DataLoader(image_raw_data, batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # ----------
    #  Training
    # ----------
    batches_done = 0
    time_s = time()
    save_csv(save_csv_loss_d, [["time", "loss"]], mode='w')
    save_csv(save_csv_loss_g, [["time", "loss"]], mode='w')

    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(data_loader):
            # imgs
            # Configure input
            real_images = imgs
            real_images.requires_grad_()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

            # Generate a batch of images
            fake_images = generator(z)

            # Real images
            real_validity = discriminator(real_images)
            # Fake images
            fake_validity = discriminator(fake_images)

            # Compute W-div gradient penalty
            real_grad_weight = Tensor(real_images.size(0), 1).fill_(1.0)
            real_grad = autograd.grad(
                real_validity, real_images, real_grad_weight, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            fake_grad_weight = Tensor(fake_images.size(0), 1).fill_(1.0)
            fake_grad = autograd.grad(
                fake_validity, fake_images, fake_grad_weight, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

            # Adversarial loss
            loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

            loss_d.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_images = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_images)
                loss_g = -torch.mean(fake_validity)

                loss_g.backward()
                optimizer_G.step()

                time_tmp = time() - time_s
                save_csv(save_csv_loss_d, [[time_tmp, loss_d.item()]], mode='a')
                save_csv(save_csv_loss_g, [[time_tmp, loss_g.item()]], mode='a')

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(data_loader), loss_d.item(), loss_g.item())
                )

                if batches_done % opt.sample_interval == 0:
                    save_image(fake_images.data[:64],
                               save_path + "%d.png" % batches_done,
                               nrow=5, normalize=True)

                batches_done += opt.n_critic


