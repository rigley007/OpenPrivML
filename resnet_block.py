import torch.nn as nn

# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        """
        Initialize the ResNet block.
        
        Args:
            dim (int): Number of input/output channels.
            padding_type (str): Type of padding ('reflect', 'replicate', or 'zero').
            norm_layer (nn.Module): Normalization layer to use (e.g., BatchNorm2d).
            use_dropout (bool): Whether to include a dropout layer in the block.
            use_bias (bool): Whether the convolutional layers should use bias.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Build the sequence of layers for the convolutional block.
        
        Args:
            dim (int): Number of input/output channels.
            padding_type (str): Type of padding ('reflect', 'replicate', or 'zero').
            norm_layer (nn.Module): Normalization layer to use.
            use_dropout (bool): Whether to include a dropout layer.
            use_bias (bool): Whether the convolutional layers should use bias.

        Returns:
            nn.Sequential: A sequential container for the layers of the convolutional block.
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # Add a convolutional layer, normalization, and activation function (ReLU)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # Add the second convolutional layer and normalization
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
         """
        Forward pass for the ResNet block.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection.
        """
        # Add the input (residual connection) to the output of the convolutional block
        out = x + self.conv_block(x)
        return out
