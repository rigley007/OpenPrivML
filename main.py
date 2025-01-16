import torch
import config as cfg
from imagenet10_dataloader import get_data_loaders
from adv_image import Adv_Gen
from regular_generator import conv_generator, Generator
from pre_model_extractor import model_extractor

if __name__ == '__main__':

    print("CUDA Available: ", torch.cuda.is_available())

    # Set device to GPU if available and enabled in config, else use CPU
    device = torch.device("cuda:0" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")

    # Load training and validation data using predefined loaders
    train_loader, val_loader = get_data_loaders()

    # Initialize feature extractor based on ResNet-18 architecture
    feature_ext = model_extractor('resnet18', 5, True)

    # Initialize the generator model
    generator = conv_generator()
    # Two different auto-encoders are provided here
    # generator = Generator(3,3)
    advGen = Adv_Gen(device, feature_ext, generator)

    # Train the adversarial generator using the training data loader and the configured number of epochs
    advGen.train(train_loader, cfg.epochs)
