# import torch.nn as nn
# from resnet_block import ResnetBlock
# from pre_model_extractor import model_extractor


# class Generator(nn.Module):
#     def __init__(self,
#                  gen_input_nc,
#                  image_nc,
#                  ):
#         super(Generator, self).__init__()

#         encoder_lis = [
#             # MNIST:1*28*28
#             nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # 8*26*26
#             nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # 16*12*12
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
#             nn.InstanceNorm2d(32),
#             nn.ReLU(),
#             # 32*5*5
#         ]

#         bottle_neck_lis = [ResnetBlock(32),
#                        ResnetBlock(32),
#                        ResnetBlock(32),
#                        ResnetBlock(32),]

#         decoder_lis = [
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.InstanceNorm2d(16),
#             nn.ReLU(),
#             # state size. 16 x 11 x 11
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.InstanceNorm2d(8),
#             nn.ReLU(),
#             # state size. 8 x 23 x 23
#             nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
#             nn.Tanh()
#             # state size. image_nc x 28 x 28
#         ]

#         self.encoder = nn.Sequential(*encoder_lis)
#         self.bottle_neck = nn.Sequential(*bottle_neck_lis)
#         self.decoder = nn.Sequential(*decoder_lis)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.bottle_neck(x)
#         x = self.decoder(x)
#         return x


# class conv_generator(nn.Module):
#     def __init__(self):
#         super(conv_generator, self).__init__()

#         self.encoder = model_extractor('resnet18', 5, True)

#         decoder_lis = [
#             ResnetBlock(64),
#             ResnetBlock(64),
#             ResnetBlock(64),
#             nn.UpsamplingNearest2d(scale_factor=2),
#             nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
#             nn.Tanh()
#             # state size. image_nc x 224 x 224
#         ]

#         self.decoder = nn.Sequential(*decoder_lis)

#     def forward(self, x):
#         x = self.encoder(x)
#         out = self.decoder(x)
#         return out


import torch.nn as nn
from resnet_block import ResnetBlock

from pre_model_extractor import model_extractor

class Generator(nn.Module):
    """Standard Generator architecture with encoder-decoder structure.
    
    This generator follows a classic encoder-decoder architecture with:
    - Encoder: Series of strided convolutions
    - Bottleneck: Multiple ResNet blocks
    - Decoder: Series of transposed convolutions
    
    Particularly designed for image-to-image translation tasks.
    """
    
    def __init__(self, gen_input_nc, image_nc):
        """Initialize the generator network.
        
        Args:
            gen_input_nc (int): Number of input channels
            image_nc (int): Number of output channels
        """
        super(Generator, self).__init__()
        
        # Encoder layers: Progressively reduce spatial dimensions while increasing channels
        encoder_lis = [
            # Input layer: gen_input_nc channels -> 8 channels
            # Input size: 28x28 -> 26x26
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            
            # Second layer: 8 channels -> 16 channels
            # Size: 26x26 -> 12x12
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            
            # Third layer: 16 channels -> 32 channels
            # Size: 12x12 -> 5x5
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        ]
        
        # Bottleneck: 4 ResNet blocks for processing features
        bottle_neck_lis = [
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32),
            ResnetBlock(32),
        ]
        
        # Decoder layers: Progressively increase spatial dimensions while decreasing channels
        decoder_lis = [
            # First upsampling: 32 channels -> 16 channels
            # Size: 5x5 -> 11x11
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            
            # Second upsampling: 16 channels -> 8 channels
            # Size: 11x11 -> 23x23
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            
            # Final layer: 8 channels -> image_nc channels
            # Size: 23x23 -> 28x28
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()  # Normalize output to [-1, 1]
        ]
        
        # Create sequential modules
        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        
    def forward(self, x):
        """Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Generated output tensor
        """
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x

class conv_generator(nn.Module):
    """Convolutional Generator using ResNet features.
    
    This generator uses a pretrained ResNet18 as encoder and
    a custom decoder with ResNet blocks and upsampling layers.
    Designed for higher resolution image generation (224x224).
    """
    
    def __init__(self):
        """Initialize the convolutional generator network."""
        super(conv_generator, self).__init__()
        
        # Use pretrained ResNet18 (first 5 layers) as encoder
        self.encoder = model_extractor('resnet18', 5, True)
        
        # Decoder architecture
        decoder_lis = [
            # ResNet blocks for processing features
            ResnetBlock(64),
            ResnetBlock(64),
            ResnetBlock(64),
            # Upsampling layers
            nn.UpsamplingNearest2d(scale_factor=2),
            # Final convolution to generate RGB image
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, 
                             output_padding=1, bias=False),
            nn.Tanh()  # Normalize output to [-1, 1]
        ]
        self.decoder = nn.Sequential(*decoder_lis)
        
    def forward(self, x):
        """Forward pass through the generator.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Generated output tensor
        """
        x = self.encoder(x)
        out = self.decoder(x)
        return out
