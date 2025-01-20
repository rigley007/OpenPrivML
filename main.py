import torch  # Import PyTorch library
import config as cfg  # Import configuration settings
from imagenet10_dataloader import get_data_loaders  # Import function to get data loaders
from adv_image import Adv_Gen  # Import adversarial image generator class
from regular_generator import conv_generator, Generator  # Import generator classes
from pre_model_extractor import model_extractor  # Import model extractor function

if __name__ == '__main__':  # Main entry point of the script

    print("CUDA Available: ", torch.cuda.is_available())  # Print if CUDA is available
    device = torch.device("cuda:0" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu")  # Set device to CUDA if available and configured, otherwise CPU

    train_loader, val_loader = get_data_loaders()  # Get training and validation data loaders
    if train_loader is None:
        raise ValueError("Error: train_loader is empty. Check dataset path or loading method.")
    if val_loader is None:
        raise ValueError("Error: val_loader is empty. Check dataset path or loading method.")


    feature_ext = model_extractor('resnet18', 5, True)  # Extract features using ResNet18 model

    generator = conv_generator()  # Initialize convolutional generator
    # Two different auto-encoders are provided here
    #generator = Generator(3,3)  # Alternative generator initialization
    advGen = Adv_Gen(device, feature_ext, generator)  # Initialize adversarial generator with device, feature extractor, and generator

    advGen.train(train_loader, cfg.epochs)  # Train adversarial generator with training data and number of epochs from config
