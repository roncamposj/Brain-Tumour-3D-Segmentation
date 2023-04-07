import torch
import torch.nn as nn 

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):

    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4: # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2)
                            ]
        if encoder_layer.dim() == 5: #3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2) : ((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2) : ((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2) : ((ds[2] + es[2]) // 2)
                            ]
    return encoder_layer, decoder_layer
        


def conv_layer(dim: int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d

def get_conv_layer(in_channels: int, 
                   out_channels: int, 
                   kernel_size: int = 3, 
                   stride: int = 1, 
                   padding: int = 1, 
                   bias: bool = True, 
                   dim: int = 2):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, 
                           stride=stride, padding=padding, bias=bias)

def conv_transpose_layer(dim: int):
    if dim == 3:
        return nn.ConvTranspose3d
    if dim == 2:
        return nn.ConvTranspose2d


def get_up_layer(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 dim: int = 3,
                 up_mode: str = 'transposed'):
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size, stride)

def maxpool_layer(dim):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    
def get_maxpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0,
                      dim: int = 2):
    return maxpool_layer(dim)(kernel_size, stride, padding)

def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()
    
def get_normalization(normalization: str,
                      num_channels: int,
                      dim: int):
    if normalization == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        
    elif normalization == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        if dim == 2:
            return nn.InstanceNorm2d(num_channels)
        
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    
class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)
        return x



class DownBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 pooling: bool = True, 
                 activation: str = 'relu', 
                 normalization: str = None, 
                 dim: str = 2, 
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        print(out_channels)
        print(self.out_channels)
        self.pooling = pooling,
        self.normalization = normalization
        
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        
        self.dim = dim
        self.activation = activation

        print('gets here0')
        # print(self.in_channels, self.out_channels,' ', 3,'', 1, self.padding, True, self.dim)
        # print(type(self.in_channels), type(self.out_channels),' ', type(3),'', type(1), 
        #       type(self.padding), type(True), type(self.dim))
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding, bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding, bias=True, dim=self.dim)
        
        print('gets here1')

        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)
        
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)
        print('gets here2')

        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            
    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling

class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 3,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode


        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)
        
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, 
                                    padding=0, bias=True, dim=self.dim)
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, 
                                    padding=self.padding, bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, 
                                    padding=self.padding, bias=True, dim=self.dim)
        

        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):

        up_layer = self.up(decoder_layer)
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)

        if self.up_mode != 'transposed':
            up_layer = self.conv0(up_layer)

        up_layer = self.act0(up_layer)

        if self.normalization:
            up_layer = self.norm0(up_layer)

        merged_layer = self.concat(up_layer, cropped_encoder_layer)
        y = self.conv1(merged_layer)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        
        y = self.conv2(y)
        y = self.act2(y)

        if self.normalization:
            y = self.norm2(y)
        
        return y




class Unet_3D(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 n_blocks: int = 4,
                 start_filters: int = 32,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            print('filters in')
            print(type(num_filters_in))
            num_filters_out = self.start_filters * (2 ** i)
            print('filters out')
            print(type(num_filters_out))
            print((num_filters_out))
            pooling = True if i < self.n_blocks - 1 else False
            print('pool')
            print(type(pooling))
            
            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   dim=self.dim,
                                   conv_mode=self.conv_mode)   

            self.down_blocks.append(down_block) 


        for i in range(self.n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.normalization,
                               conv_mode=self.conv_mode,
                               dim=self.dim,
                               up_mode=self.up_mode)
            
            self.up_blocks.append(up_block)

        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, padding=0,
                                         bias=True, dim=self.dim)    
        
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_,
                              kwargs_weights={}, kwargs_bias={}):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)
            self.bias_init(module, method_bias, **kwargs_bias)

    def forward(self, x: torch.tensor):
        encoder_output = []

        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)
        
        x = self.conv_final(x)

        return x
    
    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_'
                      not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
