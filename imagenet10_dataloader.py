import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config

def get_data_loaders():

    """
    Prepares and returns DataLoader objects for training and validation sets
    of ImageNet-10 (a subset of ImageNet with 10 classes).
    
    Returns:
        tuple: (train_loader, val_loader) - DataLoader objects for training and validation
    """

    print('==> Preparing Imagenet 10 class data..')
    # Data loading code
    traindir = config.imagenet10_traindir  # Training data directory
    valdir = config.imagenet10_valdir      # Validation data directory

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


def get_phydata_loaders():
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
