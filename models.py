import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils


class ModifAE(nn.Module):
    def __init__(self, num_traits, image_size=128, repeat_num=4, conv_dim=16):
        super(ModifAE, self).__init__()
        self.trait_num = num_traits
        self.encoder = StarganEncoder(conv_dim=conv_dim, repeat_num=repeat_num, with_dropout=False)
        self.decoder = StarganDecoder(conv_dim=conv_dim, repeat_num=repeat_num)
        self.trait_encoder = TraitEncoder(num_traits, image_size=image_size, conv_dim=conv_dim, repeat_num=repeat_num)
        self.dropout = nn.Dropout(.4)
        self.is_cuda = False

    def autoencoder_parameters(self):
        for param in self.encoder.parameters():
            yield param
        for param in self.decoder.parameters():
            yield param

    def trait_encoder_parameters(self):
        return self.trait_encoder.parameters()

    def cuda(self, device=None):
        self.is_cuda = True
        return super().cuda(device)

    def cpu(self):
        self.is_cuda = False
        return super().cpu()

    def verify_cuda(self, *tensors):
        if self.is_cuda:
            return map(lambda x: x.cuda(), tensors)
        return tensors

    def autoencode(self, img):
        z = self.encoder(img)
        return self.decoder(z)

    def forward(self, img, label):
        z = self.encoder(img)
        label_z = self.trait_encoder(label)
        z_drop = self.dropout(z)
        z_nonzeros = ((z_drop > 0) + 1).float()
        z_comb = (z_drop + label_z)/z_nonzeros
        return self.decoder(z_comb), z, label_z

    def modify(self, img, label):
        modified, _, _ = self.forward(img, label)
        return modified

    def save_autoencode_to_file(self, img, path):
        ae_img = self.autoencode(img)
        torchvision.utils.save_image(ae_img, path)

    def save_1d_traversal_to_file(self, img, label, path):
        batch_img = img.repeat(11, 1, 1, 1)
        label_traversals = torch.tensor([[float(i)]*self.trait_num for i in np.linspace(-1,1,10)])
        batch_label = torch.cat((label.cpu(), label_traversals))
        batch_img, batch_label = self.verify_cuda(batch_img, batch_label)
        out_img = self.modify(batch_img, batch_label)
        out_img[0, :, :, :] = batch_img[0, :, :, :]
        torchvision.utils.save_image(out_img, path, nrow=11)

    def save_1d_transformation_map(self, path):
        batch_label = torch.tensor([[float(i)]*self.trait_num for i in np.linspace(-1,1,10)])
        if self.is_cuda:
            batch_label = batch_label.cuda()
        torchvision.utils.save_image(self.decoder(self.trait_encoder(batch_label)), path, nrow=10)

    def save_2d_grid_traversal_to_file(self, img, num_traits, dim, path):
        batch_img = img.repeat(dim*dim+1, 1, 1, 1)
        batch_label = torch.FloatTensor([0,0]).repeat(dim*dim+1, 1)
        for i,i_val in zip(range(dim),np.linspace(-1,1,dim)):
            for j,j_val in zip(range(dim),np.linspace(-1,1,dim)):
                batch_label[i*dim+j] = torch.FloatTensor([i_val,j_val])
        batch_img, batch_label = self.verify_cuda(batch_img, batch_label)
        out_img = self.modify(batch_img, batch_label)
        out_img[-1, :, :, :] = batch_img[0, :, :, :] # Last image is original
        torchvision.utils.save_image(out_img, path, nrow=dim)

    def save_trait_interpolation_map(self, tfrom, tto, steps, path):
        ranges = list(zip(tfrom, tto))
        interps = [np.linspace(tup[0], tup[1], steps) for tup in ranges]
        spaced_ratings = list(zip(interps[0], interps[1]))
        batch_label = torch.FloatTensor(spaced_ratings).cuda()
        out_img = self.decoder(self.trait_encoder(batch_label))
        torchvision.utils.save_image(out_img, path, nrow=steps)

    def save_linear_image_interpolation(self, img, tfrom, tto, steps, path):
        batch_img = img.repeat(steps, 1, 1, 1)
        ranges = list(zip(tfrom, tto))
        interps = [np.linspace(tup[0], tup[1], steps) for tup in ranges]
        if (len(interps) > 1):
            spaced_ratings = list(zip(interps[0], interps[1]))
        else:
            spaced_ratings = interps[0].tolist()
            spaced_ratings = [[x] for x in spaced_ratings]
        batch_label = torch.FloatTensor(spaced_ratings).cuda()
        if self.is_cuda:
            batch_label = batch_label.cuda()
            batch_img = batch_img.cuda()
        out_img = self.modify(batch_img, batch_label)
        torchvision.utils.save_image(out_img, path, nrow=steps)

    def save_single_trait_modification_many_images(self, img, values, path_root):
        nimg = len(values)
        batch_img = img.repeat(nimg, 1, 1, 1)
        batch_label = torch.FloatTensor(values)
        batch_img, batch_label = self.verify_cuda(batch_img, batch_label)
        out_img = self.modify(batch_img, batch_label)
        for it in range(nimg):
            img = out_img[it]
            path = path_root + '_' + str(values[it][0]) + '.png'
            torchvision.utils.save_image(img, path)

    def save_2d_grid_interpolation(self, dim, path):
        batch_img = img.repeat(dim*dim+1, 1, 1, 1)
        batch_label = torch.FloatTensor([0,0]).repeat(dim*dim+1, 1) # 0,0 is arbitrary (doesnt matter)
        for i,i_val in zip(range(dim),np.linspace(-1,1,dim)):
            for j,j_val in zip(range(dim),np.linspace(-1,1,dim)):
                batch_label[i*dim+j] = torch.FloatTensor([i_val,j_val])
        batch_img, batch_label = self.verify_cuda(batch_img, batch_label)
        torchvision.utils.save_image(self.decoder(self.trait_encoder(batch_label)), path, nrow=dim)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class StarganEncoder(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6, with_dropout=True):
        super(StarganEncoder, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        if with_dropout:
            layers.append(nn.Dropout2d())

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            if with_dropout:
                layers.append(nn.Dropout2d())
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class StarganDecoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6):
        super(StarganDecoder, self).__init__()

        layers = []
        curr_dim = conv_dim * 4

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class TraitEncoder(nn.Module):
    def __init__(self, num_traits, image_size=128, conv_dim=64, repeat_num=6):
        super(TraitEncoder, self).__init__()
        self.target_channels = conv_dim * 4
        self.target_shape = int(image_size / 4)
        target_size = self.target_channels * self.target_shape * self.target_shape
        layers = []
        layers.append(nn.Linear(num_traits, target_size))
        layers.append(nn.LeakyReLU(0.01))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(-1, self.target_channels, self.target_shape, self.target_shape)


if __name__ == "__main__":
    img_size = 128
    repeat_num = 6
    num_traits = 2
    a = Encoder(img_size, repeat_num)
    b = Decoder(img_size, repeat_num)
    c = TraitEncoder(num_traits, img_size, repeat_num)
    ex = torch.zeros((1, 3, img_size, img_size))
    ex2 = torch.zeros((1, num_traits))
    print(a(ex).shape)
    print(b(a(ex)).shape)
    print(c(ex2).shape)
    d = ModifAE(num_traits, img_size, repeat_num)
    print(d(ex, ex2))
