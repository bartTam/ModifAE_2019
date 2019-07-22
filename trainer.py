import torch.optim
import torch.nn as nn
import os.path
import tqdm

import models
from config import config

learning_rate = 0.0008
class Trainer:
    def __init__(self, data_loader, total_epochs, current_epoch=None):
        self.data_loader = data_loader
        self.total_epochs = total_epochs
        self.starting_epoch = 0

        # Create the model and load from saved state (if given)
        self.model = models.ModifAE(len(config.targets), image_size=config.input_image_size, repeat_num=config.repeat_num, conv_dim=config.conv_dim)
        print("Total number of parameters in use:", sum(p.numel() for p in self.model.parameters()))

        if config.continue_training:
            self.model.load(os.path.join(config.log_path, "model"))
            self.starting_epoch = config.current_epoch

        if torch.cuda.is_available():
            self.model.cuda()

        self.autoencoder_optimizer = torch.optim.Adam(self.model.autoencoder_parameters(), lr=learning_rate)
        self.trait_encoder_optimizer = torch.optim.Adam(self.model.trait_encoder_parameters(), lr=learning_rate)
        self.autoencoder_criterion = nn.L1Loss()
        self.trait_criterion = nn.L1Loss()

        self.fixed_image, self.fixed_label = data_loader.get_fixed_example()
        self.fixed_image, self.fixed_label = self.model.verify_cuda(self.fixed_image, self.fixed_label)
        self.fixed_image = self.fixed_image.view([1] + list(self.fixed_image.shape))
        self.fixed_label = self.fixed_label.view([1] + list(self.fixed_label.shape))

    def train(self):
        with tqdm.tqdm(range(self.starting_epoch, self.total_epochs)) as loading_bar:
            for epoch_num in loading_bar:
                for batch_number, (image, label) in enumerate(self.data_loader):
                    #image = self.model.normalize_image(image.cuda())
                    image, label = self.model.verify_cuda(image, label)
                    ae_image, z, z_prime = self.model(image, label)

                    self.autoencoder_optimizer.zero_grad()
                    self.trait_encoder_optimizer.zero_grad()

                    # AE Loss
                    autoencoder_loss = self.autoencoder_criterion(ae_image, image)
                    autoencoder_loss.backward(retain_graph=True)
                    self.autoencoder_optimizer.step()

                    # Trait loss
                    z = z.detach()
                    self.trait_encoder_optimizer.zero_grad() # This makes trait net independent
                    trait_loss = self.trait_criterion(z_prime, z)
                    trait_loss.backward()
                    self.trait_encoder_optimizer.step()

                    loading_bar.set_postfix(epoch=epoch_num, ae_loss=autoencoder_loss.data, trait_loss=trait_loss.data)

                    if batch_number % 1000 == 0:
                        # Debug AE batches
                        self.model.save_autoencode_to_file(self.fixed_image, os.path.join(config.log_path, 
                                        "{}_AE_Batch{}.png".format(epoch_num, batch_number)))
                        self.model.save_1d_traversal_to_file(self.fixed_image, self.fixed_label, os.path.join(config.log_path, 
                                    "{}_T_Batch{}.png".format(epoch_num, batch_number)))
                        self.model.save_1d_transformation_map(os.path.join(config.log_path, 
                                    "{}_Grid_traits_Batch{}.png".format(epoch_num, batch_number)))

                # Save examples each epoch
                self.model.save_1d_transformation_map(os.path.join(config.log_path, "{}_Grid_traits.png".format(epoch_num)))
                self.model.save_autoencode_to_file(self.fixed_image, os.path.join(config.log_path, "{}_AE.png".format(epoch_num)))
                self.model.save_1d_traversal_to_file(self.fixed_image, self.fixed_label, os.path.join(config.log_path, "{}_T.png".format(epoch_num)))

                self.model.save(os.path.join(config.log_path, "model"))
