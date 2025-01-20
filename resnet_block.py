import torch.nn as nn

# Define a ResNet block class (modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py)
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
        
        # Print out parameters during initialization
        print(f"Initializing ResNet block with {dim} channels, padding type: {padding_type}, use_dropout: {use_dropout}, use_bias: {use_bias}")
        
        # Build the convolutional block
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
        print(f"Building convolutional block with padding: {padding_type}")

        conv_block = []
        p = 0
        
        # Choose padding based on the specified type
        if padding_type == 'reflect':
            print("Using ReflectionPad2d for padding.")
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            print("Using ReplicationPad2d for padding.")
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
            print("Using zero padding.")
        else:
            raise NotImplementedError(f'Padding type [{padding_type}] is not implemented.')

        # First convolutional layer with normalization and ReLU activation
        print("Adding first convolutional layer, normalization, and ReLU activation.")
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        # Add dropout layer if specified
        if use_dropout:
            print("Adding dropout layer with p=0.5.")
            conv_block += [nn.Dropout(0.5)]

        # Second convolutional layer
        p = 0
        if padding_type == 'reflect':
            print("Using ReflectionPad2d for padding in second layer.")
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            print("Using ReplicationPad2d for padding in second layer.")
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
            print("Using zero padding in second layer.")
        else:
            raise NotImplementedError(f'Padding type [{padding_type}] is not implemented.')

        # Second convolutional layer with normalization
        print("Adding second convolutional layer and normalization.")
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        # Return the sequential block
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Forward pass for the ResNet block.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual connection.
        """
        # Print input size before passing through the block
        print(f"Forward pass input shape: {x.shape}")

        # Apply the convolution block and add the input tensor to the output (residual connection)
        out = x + self.conv_block(x)

        # Print output shape after residual connection
        print(f"Forward pass output shape: {out.shape}")
        return out
