# General Configuration
use_cuda = True

# Number of image channels (3 for RGB images)
image_nc = 3

# Number of training epochs
epochs = 800

# Batch size for training
batch_size = 64

# Minimum and maximum values for bounding boxes
BOX_MIN = 0
BOX_MAX = 1

# Pretrained model architecture to use (ResNet-18)
pretrained_model_arch = 'resnet18'

# Number of layers to extract features from
num_layers_ext = 5

# Whether to keep the feature extractor layers fixed during training
ext_fixed = True

# Whether to tag generated images
G_tagged = False

# Size of the tags to be added to generated images
tag_size = 6

# Coefficient for noise to be added to images
noise_coeff = 0.35

# Whether to concatenate generated images with tags
cat_G = False

# Whether to add noise to images
noise_img = True


# Path to the pre-trained generator model

noise_g_path = './models/netG_epoch_160.pth'

# Path to the pre-trained generator model without tags
noTag_noise_g_path = './models/noTag_netG_epoch_80.pth'


# Directory for ImageNet-10 training images
imagenet10_traindir = '~/Pictures/transfer_imgnet_10/train'

# Directory for ImageNet-10 validation images
imagenet10_valdir = '~/Pictures/transfer_imgnet_10/val'

# Directory for ImageNet-10 physical validation images
imagenet10_phyvaldir = '~/Pictures/phy/val'


=======
# Path to save models
models_path = './models/'

# Path to save adversarial images
adv_img_path = './images/'

# Path to save CIFAR-10 models
cifar10_models_path = './models/'

# Path to save CIFAR-10 adversarial images
cifar10_adv_img_path = './images/0828/adv/'

# Use Automatic Mixed Precision (AMP) for training
use_amp = True
