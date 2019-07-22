import os
import argparse
import loader
import models
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='/raid/SAGAN/CelebA/images/')
parser.add_argument('--first_img_num', type=int, required=True)
parser.add_argument('--num_traversals', type=int, default=4)
parser.add_argument('--input_image_size', type=int, default=128)
parser.add_argument('--log_path', type=str, required=True)
parser.add_argument('--num_traits', type=int, required=True)
parser.add_argument('--grid_dim', type=int, default=4, help="Dimension of grid traversal")
config = parser.parse_args()

dataloader = loader.Traverse_Dataloader(config.image_path,
                                      config.first_img_num,
                                      config.num_traversals,
                                      config.input_image_size,
                                      None)

model = models.ModifAE(config.num_traits, image_size=config.input_image_size)
model.load(os.path.join(config.log_path, "model"))
model.cuda()
print("Total number of parameters in use:", sum(p.numel() for p in model.parameters()))

tracker = 0
for img_number, (img, img_path) in enumerate(dataloader):
    if tracker >= config.num_traversals:
        break

    # Generate grid traverse
    if config.num_traits >= 2:
        save_path = str(config.first_img_num + img_number) + '_grid_traverse.png'
        model.save_2d_grid_traversal_to_file(img, config.num_traits, config.grid_dim, save_path)

    # Generate linear traversal
    fake_label = torch.FloatTensor([np.zeros(config.num_traits)]).cuda()
    save_path = str(config.first_img_num + img_number) + '_linear_traverse.png'
    model.save_1d_traversal_to_file(img, fake_label, save_path)

    # Generate multiple modified images
    if config.num_traits == 1:
        path_root = str(config.first_img_num + img_number)
        model.save_single_trait_modification_many_images(img, [[-1.],[0.0],[1.]], path_root)

    # Generate images distributed over trait range
    if config.num_traits == 1: # Can manually modify this to work for n traits
        steps = 4
        r_from = [-1]  # can define this as [-1,-1] for example to interpolate over two traits
        r_to = [1]  # can define this as [1,1] for example to interpolate over two traits
        save_path = str(config.first_img_num + img_number) + '_' + config.log_path[4:-1] + '.png'
        model.save_linear_image_interpolation(img, r_from, r_to, steps, save_path)
