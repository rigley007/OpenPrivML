import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config

# Function to prepare data loaders for the Imagenet 10 dataset
def get_data_loaders():
    """
    Prepares and returns data loaders for training and validation datasets
    for the Imagenet 10 class dataset.

    :return: train_loader, val_loader
    """
    print('==> Preparing Imagenet 10 class data..')
    # Data loading code
    traindir = config.imagenet10_traindir
    valdir = config.imagenet10_valdir

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=True,
        num_workers=12, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=True,
        num_workers=12, pin_memory=True)

    return train_loader, val_loader

# Function to prepare data loaders for the physical dataset
def get_phydata_loaders():
    """
    Prepares and returns a data loader for a physical validation dataset
    for the Imagenet 10 class dataset.

    :return: val_loader
    """
    print('==> Preparing Physical Imagenet 10 class data..')
    # Data loading code

    valdir = config.imagenet10_phyvaldir

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=True,
        num_workers=12, pin_memory=True)

    return val_loader
