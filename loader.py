import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import os.path
import torchvision.transforms as transforms
import torch
from random import shuffle

class CelebA_Dataloader(data.DataLoader):
    def __init__(self, csv_path, images_path, targets, batch_size, num_workers,
                 image_size, transform=transforms.ToTensor):
        super(CelebA_Dataloader, self).__init__(
            CelebA_Dataset(csv_path, images_path, targets,
                           transforms.Compose([
                               transforms.Resize((image_size, image_size)),
                               transform()
                           ])), batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=int(num_workers))
        self.number_of_targets = len(targets)

    def get_fixed_example(self):
        img, label = self.dataset[0]
        return img, label

class CelebA_Dataset(data.Dataset):
    def __init__(self, csv_path, images_path, targets, transform=None):
        self.images_path = images_path
        self.transform = transform

        # Read the attributes
        attributes_df = pd.read_csv(csv_path)
        attributes_df = attributes_df.groupby(['Filename']).mean()
        self.targeted_attributes_df = attributes_df[targets]
        self.targeted_attributes_df.reset_index(level=0, inplace=True)

        
        for target in targets:
            col = self.targeted_attributes_df[target]
            # If binary trait
            if col.max() == 1:
                norm = col - .5
            # 1 - 9 continuous trait
            else:
                norm = (col-5)/4.0
            pd.set_option('mode.chained_assignment', None)
            self.targeted_attributes_df.loc[target] = norm
        self.targeted_attributes_df = self.targeted_attributes_df.dropna()

    def normalize(self, data):
        return data

    def __getitem__(self, index):
        item = self.targeted_attributes_df.iloc[index]
        filename = item[0]
        targets = item[1:]

        # Get the targets
        normalized_targets = torch.from_numpy(self.normalize(np.array(targets, dtype='float')))
        
        # Get the image
        path = os.path.join(self.images_path, filename)
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image.float(), normalized_targets.float()

    def __len__(self):
        return len(self.targeted_attributes_df)

#----------------------------------------------------------------------------------------

class Traverse_Dataloader(data.DataLoader):
    def __init__(self, images_path, first_img_int, n_imgs, image_size, imgnames=None, transform=transforms.ToTensor):
        super(Traverse_Dataloader, self).__init__(
            Traverse_Dataset(images_path, first_img_int, n_imgs, imgnames, transforms.Compose([ transforms.Resize((image_size, image_size)), transform() ])))

class Traverse_Dataset(data.Dataset):
    def __init__(self, images_path, first_img_int, n_imgs, imgnames=None, transform=None):
        self.images_path = images_path
        self.transform = transform
        # Grab sequential image numbers (6 digit image names from CelebA)
        self.imgs = [images_path + str(int(first_img_int)+i).zfill(6) + '.jpg' for i in range(n_imgs)]
        if imgnames:
            imgnames = pd.read_csv(imgnames, header=None)[0].tolist()
            shuffle(imgnames)
            self.imgs = [images_path + imgname for imgname in imgnames]


    def __getitem__(self, index):
        path = self.imgs[index]
        
        # Get the image
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image.float(), path

    def __len__(self):
        return len(self.imgs)
