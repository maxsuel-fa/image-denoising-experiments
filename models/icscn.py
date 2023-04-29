import numpy as np
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.cuda import device_count
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from typing import Any, Dict, Union
from collections import OrderedDict

from .losses import AdvLoss, DiscLoss, FMLoss, PerceptualLoss
from .networks import Discriminator, Generator
from data.dataset import Dataset
from utils.options import parse_train_options

import matplotlib.pyplot as plt

from visualization.visualization import plot_result
from torchvision.transforms import ToPILImage, Resize
from PIL import Image, ImageChops, ImageOps

class ICSCN:
    """ A de-raining model to remove rain droplets from images.
    To do such task, this model is trained against a PatchGAN discriminator
    in a generative adversarial network style.

    The name ICSCN extends to 'I Can See Clearly Now', the name of the paper
    where this model was presented.

    You can find the paper in: https://arxiv.org/abs/1901.00893
    """
    def __init__(self,
                 image_nc: int = 3,
                 is_train: bool = False) -> None:
        """ Constructs a new de-raining network.

        Parameters:
            image_nc (int) -- the number of channels of the image given as
                              input to the model. Default: 3
            is_train (bool) -- whether to train the model or not. Default: False
        """
        self.model = Generator(image_nc, image_nc)

        if is_train:
            # The input of conditinal GANs' discriminator is the generated data
            # concatenaited with the real data. Thus disc_input_nc = 2 * image_nc
            self.discriminator = Discriminator(image_nc)

            # Initializes the weghts of both generator and discriminator
            self.model.apply(self._weights_init)
            self.discriminator.apply(self._weights_init)

    def __call__(self, x: Tensor) -> Tensor:
        """ Makes the model callable by generating a clear image.

        Parameters:
            x (Tensor) -- rainy input image
        Return (Tensor) -- the clear version of the input image
        """
        return self.model(x)

    def train(self,
              train_dataset: Dataset,
              train_options: Any) -> None:
        """  Given a train dataset and training options, trains a ICSCN
        de-raining network in such dataset.

        Parameters:
            train_dataset (Dataset) -- the dataset to be used in the training
            train_options (Any) -- options like number of gpus to be used,
                                   batch size, learning rate and so on
        """
        dist.init_process_group('nccl')

        rank = dist.get_rank()
        n_gpus = train_options.n_gpus

        assert (n_gpus <= device_count()
                and n_gpus <= train_options.batch_size)

        sampler = DistributedSampler(train_dataset,
                                     num_replicas=n_gpus,
                                     rank=rank)

        dataloader = DataLoader(train_dataset,
                                shuffle=False,
                                batch_size=train_options.batch_size,
                                num_workers=0,
                                sampler=sampler)

        # distributed models
        device_id = 1
        self.model.to(device_id)
        self.ddp_gen = DDP(self.model, device_ids=[device_id])
        self.discriminator.to(device_id)
        self.ddp_disc = DDP(self.discriminator, device_ids=[device_id])

        # optimizers
        self.gen_optim = optim.Adam(self.ddp_gen.parameters(), lr=train_options.lr)
        self.disc_optim = optim.Adam(self.ddp_disc.parameters(), lr=train_options.lr)

        # generator criterions
        adv_criterion = AdvLoss().to(device_id)
        fm_criterion = FMLoss().to(device_id)
        vgg_criterion = PerceptualLoss([1, 6, 11, 20, 29]).to(device_id)

        # discriminator criterion
        disc_criterion = DiscLoss().to(device_id)

        if train_options.start_epoch > 0:
            self.load(rank,
                      checkpoint_dir=train_options.checkpoint_dir,
                      epoch=train_options.start_epoch,
                      is_train=True)

        start_time = time.time()
        to_pil = ToPILImage(mode='RGB')
        figure = plt.figure(figsize=(20, 5), constrained_layout=True)
        grid_rows = 1
        grid_cols = 4

        for epoch in range(train_options.start_epoch,
                           train_options.n_epochs + train_options.start_epoch):
            epoch_start_time = time.time()

            if train_options.start_decay and epoch >= train_options.start_decay:
                self._update_lr(train_options.lr,
                                epoch,
                                train_options.start_decay,
                                train_options.end_decay)

            for i, (rainy_image, clear_image) in enumerate(dataloader):
                rainy_image = rainy_image.to(device_id)
                clear_image = clear_image.to(device_id)

                # fake (de-rained) generation
                derained_image = self.ddp_gen(rainy_image)

                # concatenating the images to form the conditional input
                # for the discriminator
                #clear_clear = torch.cat((clear_image, clear_image), dim=1)
                #derained_clear = torch.cat((derained_image, clear_image),
                #dim=1)

                # updating the discriminator
                clear_pred = self.ddp_disc(clear_image)
                derained_pred = self.ddp_disc(derained_image)

                disc_loss = disc_criterion(clear_pred,
                                           derained_pred)

                self.disc_optim.zero_grad()
                disc_loss.backward(retain_graph=True)
                self.disc_optim.step()

                # updating the generator
                clear_pred = self.ddp_disc(clear_image, extract_feat=True)
                derained_pred = self.ddp_disc(derained_image, extract_feat=True)

                adv_loss = adv_criterion(derained_pred[-1])
                fm_loss = fm_criterion(clear_pred[:-2], derained_pred[:-2])
                vgg_loss = vgg_criterion(derained_image, clear_image)

                gen_loss = train_options.lambda_adv * adv_loss \
                    + train_options.lambda_fm * fm_loss \
                    + train_options.lambda_vgg * vgg_loss

                self.gen_optim.zero_grad()
                gen_loss.backward()
                self.gen_optim.step()

                """derained_image = torch.squeeze(derained_image, dim=0)
                derained_image = to_pil(derained_image)

                rainy_image = torch.squeeze(rainy_image, dim=0)
                rainy_image = to_pil(rainy_image)
        
                clear_image = torch.squeeze(clear_image, dim=0)
                clear_image = to_pil(clear_image)

                figure.add_subplot(grid_rows, grid_cols, 1)
                plt.imshow(clear_image, cmap=None)
                plt.axis('off')
                plt.title('Ground Truth')

                figure.add_subplot(grid_rows, grid_cols, 2)
                plt.imshow(rainy_image, cmap=None)
                plt.axis('off')
                plt.title('Rainy Image')

                figure.add_subplot(grid_rows, grid_cols, 3)
                plt.imshow(derained_image, cmap=None)
                plt.axis('off')
                plt.title('Derained Image')

                difference = ImageChops.difference(clear_image, derained_image).convert('L')

                figure.add_subplot(grid_rows, grid_cols, 4)
                plt.imshow(difference, cmap='gray')
                plt.axis('off')
                plt.title('Error Pixel per Pixel')

                plt.pause(1e-9)"""
            # calculates the time already spent on training
            curr_spent_time = time.time() - start_time

            # calculates the speed of the training in epochs per second
            trained_epochs = (epoch - train_options.start_epoch + 1)
            epochs_per_sec = trained_epochs / curr_spent_time

            # calculates the left time until the end of the training
            time_left = (train_options.n_epochs - trained_epochs) * (time.time() - epoch_start_time)

            time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
            curr_spent_time = time.strftime("%H:%M:%S",
                                            time.gmtime(curr_spent_time))

            # TODO: add loggers instead of simple prints
            logg_str = 'epochs: {:>10}/{:<10} time: {:>10}/{:<10}'
            logg_str = logg_str.format(trained_epochs,
                                       train_options.n_epochs,
                                       curr_spent_time,
                                       time_left)
            print(logg_str)

            # save the model each 50 epoch
            if trained_epochs and trained_epochs % 10 == 0:
                pass
                #self.save(train_options.checkpoint_dir, epoch + 1)

        self.save(train_options.checkpoint_dir)

    def save(self,
             checkpoint_dir: str,
             current_epoch: Union[int, str] = 'latest') -> None:
        """ Save the current state of both the generator and discriminator.

        Parameters:
            checkpoint_dir (str) -- path to the directory where the modules
                                    will be saved
            current_epoch (Union[int, str]) -- the current training epoch.
                                               Default: 'latest'
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_path = 'weights_' + 'epoch_' + str(current_epoch) + '.tar'
        save_path = os.path.join(checkpoint_dir, save_path)

        torch.save({'gen_state_dict': self.ddp_gen.state_dict(),
                    'disc_state_dict': self.ddp_disc.state_dict(),
                    'gen_optim_state_dict': self.gen_optim.state_dict(),
                    'disc_optim_state_dict': self.disc_optim.state_dict()},
                   save_path)

    def load(self,
             rank: int,
             checkpoint_dir: str,
             epoch: Union[int, str] = 'latest',
             is_train: bool = False) -> None:
        """ Load the last checkpoint trained until the given epoch.

        Parameters:
            rank (int) -- TODO
            checkpoint_dir (str) -- path to the directory where the model was saved
            epoch (int) -- the checkpoint epoch. Default: 'latest'
            is_train (bool) -- whether is training regime or not. Default: False
        """
        load_path = 'weights_' + 'epoch_' + str(epoch) + '.tar'
        load_path = os.path.join(checkpoint_dir, load_path)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_dict = torch.load(load_path, map_location=map_location)

        if is_train:
            self.ddp_gen.load_state_dict(checkpoint_dict['gen_state_dict'])
            self.ddp_disc.load_state_dict(checkpoint_dict['disc_state_dict'])
            self.gen_optim.load_state_dict(checkpoint_dict['gen_optim_state_dict'])
            self.disc_optim.load_state_dict(checkpoint_dict['disc_optim_state_dict'])
            return

        unwraped_state_dict = self._unwrap_state_dict(checkpoint_dict['gen_state_dict'])
        self.model.load_state_dict(unwraped_state_dict)

    def _weights_init(self, module: nn.Module) -> None:
        """ TODO """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def _unwrap_state_dict(self,
                           state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """ TODO """
        unwraped_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = k[7:]
            unwraped_state_dict[k] = v
        return unwraped_state_dict

    def _update_lr(self,
                   initial_lr: float,
                   current_epoch: int,
                   start_decay: int = 50,
                   end_decay: int = 100,
                   verbose: bool = True) -> None:
        """ TODO """
        lr = (initial_lr / (end_decay - start_decay)) * (end_decay - current_epoch)

        for param_group in self.gen_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.disc_optim.param_groups:
            param_group['lr'] = lr

        if verbose:
            print(f'The new learning rate is: {lr}')


