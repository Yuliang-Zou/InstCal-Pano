# Credit: https://github.com/hendrycks/imagenet-r/blob/master/DeepAugment/CAE_Model/cae_32x32x32_zero_pad_bin.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random


class CAE(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # a
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation
        option = np.random.choice(range(9))
        if option == 1:
            # set some weights to zero
            H = ec3.size()[2]
            W = ec3.size()[3]
            mask = (torch.cuda.FloatTensor(H, W).uniform_() > 0.2).float().cuda()
            ec3 = ec3 * mask
            del mask
        elif option == 2:
            # negare some of the weights 
            H = ec3.size()[2]
            W = ec3.size()[3]
            mask = (((torch.cuda.FloatTensor(H, W).uniform_() > 0.1).float() * 2) - 1).cuda()
            ec3 = ec3 * mask
            del mask
        elif option == 3:
            num_channels = 10
            perm = np.array(list(np.random.permutation(num_channels)) + list(range(num_channels, ec3.size()[1])))
            ec3 = ec3[:, perm, :, :]   
        elif option == 4:
            num_channels = ec3.shape[1]
            num_channels_transform = 5
            
            _k = random.randint(1,3)
            _dims = [0, 1, 2]
            random.shuffle(_dims)
            _dims = _dims[:2]

            for i in range(num_channels_transform):
                filter_select = random.choice(list(range(num_channels)))
                ec3[:,filter_select] = torch.flip(ec3[:,filter_select], dims=_dims)
        elif option == 5:
            num_channels = ec3.shape[1]
            num_channels_transform = num_channels
            
            _k = random.randint(1,3)
            _dims = [0, 1, 2]
            random.shuffle(_dims)
            _dims = _dims[:2]

            for i in range(num_channels_transform):
                if i == num_channels_transform / 2:
                    _dims = [_dims[1], _dims[0]]
                ec3[:,i] = torch.flip(ec3[:,i], dims=_dims)
        elif option == 6:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                for j in range(z.size(0)):
                    shift_x, shift_y = 1, 1# np.random.randint(3, size=(2,))
                    z[j,j,shift_x,shift_y] = 1 # np.random.choice([1.,-1.])
                
                # Without this line, z would be the identity convolution
                z = z + ((torch.rand_like(z) - 0.5) * 0.2)
                ec3 = F.conv2d(ec3, z, padding=1)
                del z
        elif option == 7:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                for j in range(z.size(0)):
                    shift_x, shift_y = 1, 1# np.random.randint(3, size=(2,))
                    z[j,j,shift_x,shift_y] = 1 # np.random.choice([1.,-1.])
                    
                    if random.random() < 0.5:
                        rand_layer = random.randint(0, c - 1)
                        z[j, rand_layer, random.randint(-1, 1), random.randint(-1, 1)] = 1
                
                ec3 = F.conv2d(ec3, z, padding=1)
                del z
        elif option == 8:
            with torch.no_grad():
                c, h, w = ec3.shape[1], ec3.shape[2], ec3.shape[3]
                z = torch.zeros(c, c, 3, 3).cuda()
                shift_x, shift_y = np.random.randint(3, size=(2,))
                for j in range(z.size(0)):
                    if random.random() < 0.2:
                        shift_x, shift_y = np.random.randint(3, size=(2,))

                    z[j,j,shift_x,shift_y] = 1 # np.random.choice([1.,-1.])

                # Without this line, z would be the identity convolution
                # z = z + ((torch.rand_like(z) - 0.5) * 0.2)
                ec3 = F.conv2d(ec3, z, padding=1)
                del z

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
        if option == 0: 
            self.encoded = self.encoded *\
            (3 + 2 * np.float32(np.random.uniform()) * (2*torch.rand_like(self.encoded-1)))
        return self.decode(self.encoded)

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec


def get_cae_weights(model_path='models/cae_model.state'):
    weight_keys = [
        'e_conv_1.1.weight', 'e_conv_1.1.bias', 'e_conv_2.1.weight', 'e_conv_2.1.bias', 'e_block_1.1.weight',
        'e_block_1.1.bias', 'e_block_1.4.weight', 'e_block_1.4.bias', 'e_block_2.1.weight', 'e_block_2.1.bias',
        'e_block_2.4.weight', 'e_block_2.4.bias', 'e_block_3.1.weight', 'e_block_3.1.bias', 'e_block_3.4.weight',
        'e_block_3.4.bias', 'e_conv_3.0.weight', 'e_conv_3.0.bias', 'd_up_conv_1.0.weight', 'd_up_conv_1.0.bias',
        'd_up_conv_1.3.weight', 'd_up_conv_1.3.bias', 'd_block_1.1.weight', 'd_block_1.1.bias', 'd_block_1.4.weight',
        'd_block_1.4.bias', 'd_block_2.1.weight', 'd_block_2.1.bias', 'd_block_2.4.weight', 'd_block_2.4.bias',
        'd_block_3.1.weight', 'd_block_3.1.bias', 'd_block_3.4.weight', 'd_block_3.4.bias', 'd_up_conv_2.0.weight',
        'd_up_conv_2.0.bias', 'd_up_conv_2.3.weight', 'd_up_conv_2.3.bias', 'd_up_conv_3.0.weight', 'd_up_conv_3.0.bias',
        'd_up_conv_3.3.weight', 'd_up_conv_3.3.bias'
    ]
    key_mapping = dict([(str(int(i / 2)) + ".weight", key) if i % 2 == 0 else (str(int(i / 2)) + ".bias", key) for i, key in enumerate(weight_keys)])
    NUM_LAYERS = int(len(key_mapping.values()) / 2) # 21
    NUM_DISTORTIONS = 8
    OPTION_LAYER_MAPPING = {0: range(11, NUM_LAYERS - 5), 1: range(8, NUM_LAYERS - 7), 2: range(8, NUM_LAYERS - 7), 3: range(10, NUM_LAYERS - 7), 4: range(8, NUM_LAYERS - 7), 5: range(8, NUM_LAYERS - 7), 6: range(8, NUM_LAYERS - 7), 7: range(8, NUM_LAYERS - 7), 8: range(8, NUM_LAYERS - 7)}

    def get_name(i, tpe):
        return key_mapping[str(i) + "." + tpe]

    weights = torch.load(model_path)
    for option in random.sample(range(NUM_DISTORTIONS), 1):
        i = np.random.choice(OPTION_LAYER_MAPPING[option])
        j = np.random.choice(OPTION_LAYER_MAPPING[option])
        weight_i = get_name(i, "weight")
        bias_i = get_name(i, "bias")
        weight_j = get_name(j, "weight")
        bias_j = get_name(j, "weight")
        if option == 0:
            weights[weight_i] = torch.flip(weights[weight_i], (0,))
            weights[bias_i] = torch.flip(weights[bias_i], (0,))
            weights[weight_j] = torch.flip(weights[weight_j], (0,))
            weights[bias_j] = torch.flip(weights[bias_j], (0,))
        elif option == 1:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(12)]:
                weights[weight_i][k] = -weights[weight_i][k]
                weights[bias_i][k] = -weights[bias_i][k]
        elif option == 2:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
                weights[weight_i][k] = 0 * weights[weight_i][k]
                weights[bias_i][k] = 0 * weights[bias_i][k]
        elif option == 3:
            for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
                weights[weight_i][k] = -gelu(weights[weight_i][k])
                weights[bias_i][k] = -gelu(weights[bias_i][k])
        elif option == 4:
            weights[weight_i] = weights[weight_i] *\
            (1 + 2 * np.float32(np.random.uniform()) * (4*torch.rand_like(weights[weight_i]-1)))
            weights[weight_j] = weights[weight_j] *\
            (1 + 2 * np.float32(np.random.uniform()) * (4*torch.rand_like(weights[weight_j]-1)))
        elif option == 5: ##### begin saurav #####
            if random.random() < 0.5:
                mask = torch.round(torch.rand_like(weights[weight_i]))
            else:
                mask = torch.round(torch.rand_like(weights[weight_i])) * 2 - 1
            weights[weight_i] *= mask
        elif option == 6:
            _k = random.randint(1, 3)
            weights[weight_i] = torch.rot90(weights[weight_i], k=_k, dims=(2,3))
        elif option == 7:
            out_filters = weights[weight_i].shape[0]
            to_zero = list(set([random.choice(list(range(out_filters))) for _ in range(out_filters // 5)]))
            weights[weight_i][to_zero] = weights[weight_i][to_zero] * -1.0
        elif option == 8:
            # Only keep the max filter value in the conv 
            c1, c2, width = weights[weight_i].shape[0], weights[weight_i].shape[1], weights[weight_i].shape[2]
            assert weights[weight_i].shape[2] == weights[weight_i].shape[3]

            w = torch.reshape(weights[weight_i], shape=(c1, c2, width ** 2))
            res = torch.topk(w, k=1)

            w_new = torch.zeros_like(w).scatter(2, res.indices, res.values)
            w_new = w_new.reshape(c1, c2, width, width)
            weights[weight_i] = w_new
        
    return weights


if __name__ == '__main__':
    net = CAE()
    model_path = '/home/zouyuliang123/research/nips2021workshop/pretrained_model/cae_model.state'
    net.load_state_dict(get_cae_weights(model_path))
    net.cuda()
    net.eval()

    from PIL import Image
    im = Image.open('/home/zouyuliang123/research/nips2021workshop/debug.jpg')
    from torchvision import transforms
    transform = transforms.ToTensor()
    im_t = transform(im).cuda()
    im_t = im_t.unsqueeze(0)

    with torch.no_grad():
        out = net(im_t)

    import cv2
    out = out[0].cpu().permute(1, 2, 0).numpy()
    cv2.imwrite('debug2.jpg', (255*out).astype(np.uint8)[:,:,::-1])
