
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import os
import config as cfg
from transfer_learning_clean_imagenet10_0721 import Imagenet10ResNet18

# Define paths for saving models and adversarial images
models_path = cfg.models_path
adv_img_path = cfg.adv_img_path

def weights_init(m):
    """Initialize network weights using specific distributions.
    
    Args:
        m (nn.Module): Neural network module to initialize
        
    This function applies custom initialization:
    - Convolutional layers: Normal distribution with mean=0.0, std=0.02
    - BatchNorm layers: Weights from N(1.0, 0.02), biases=0
    """
    classname = m.__class__.__name__
    # Check if the layer is a convolutional layer
    # The find() method returns -1 if 'Conv' is not found in the class name
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Adv_Gen:
    """Adversarial Generator class for creating adversarial images.
    
    This class implements an adversarial generator that creates perturbed images
    designed to fool a target classifier while maintaining visual similarity
    to original images.
    """
    
    def __init__(self, device, model_extractor, generator):
        """Initialize the adversarial generator.
        
        Args:
            device (torch.device): Device to run computations on (CPU/GPU)
            model_extractor (nn.Module): Model for extracting image features
            generator (nn.Module): Generator network for creating adversarial perturbations
        """


        # Assign the computation device (e.g., "cuda" for GPU or "cpu").
        self.device = device
        # Initialize the feature extractor model. This model is responsible for extracting relevant features from the input data.
        self.model_extractor = model_extractor  # Feature extractor model
        self.generator = generator  # Generator model

        self.box_min = cfg.BOX_MIN  # Minimum value for pixel normalization
        self.box_max = cfg.BOX_MAX  # Maximum value for pixel normalization

        self.ite = 0  # Iteration counter
        
        # Move models to specified device
        self.model_extractor.to(device)
        self.generator.to(device)
        
        # Setup classifier (ResNet18 pretrained on ImageNet10)
        self.classifer = Imagenet10ResNet18()
        self.classifer.load_state_dict(torch.load('models/resnet18_imagenet10_transferlearning.pth'))
        self.classifer.to(device)
        # Enable multi-GPU training for classifier
        self.classifer = torch.nn.DataParallel(self.classifer, device_ids=[0, 1])
        
        # Set classifier to eval mode but keep BatchNorm in training mode
        self.classifer.train()
        for p in self.classifer.parameters():
            p.requires_grad = False
        
        # Initialize Adam optimizer for generator
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        
        # Create necessary directories
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(adv_img_path):
            os.makedirs(adv_img_path)
            
    # define batch train function
    def train_batch(self, x):
        """Train generator on a single batch of images.
        
        Args:
            x (torch.Tensor): Batch of input images
            
        Returns:
            tuple: (adversarial loss value, generated adversarial images, classifier predictions)
        """
        # Reset gradients
        self.optimizer_G.zero_grad()
        
        # Generate adversarial images
        adv_imgs = self.generator(x)
        
        # Compute features and predictions (no gradient computation needed)
        with torch.no_grad():

            class_out = self.classifer(adv_imgs)  # Classifier output for adversarial images
            tagged_feature = self.model_extractor(x)  # Extract features from clean images

        adv_img_feature = self.model_extractor(adv_imgs)  # Extract features from adversarial images
        # Calculate adversarial loss using L1 distance between features
        # Multiply by noise coefficient to control perturbation magnitude
        # Compute adversarial loss using L1 loss between features

        loss_adv = F.l1_loss(tagged_feature, adv_img_feature * cfg.noise_coeff)
        loss_adv.backward(retain_graph=True)
        
        # Update generator
        self.optimizer_G.step()
        
        return loss_adv.item(), adv_imgs, class_out

    def train(self, train_dataloader, epochs):
        """Train the generator for multiple epochs.
        
        Args:
            train_dataloader (DataLoader): DataLoader for training data
            epochs (int): Number of epochs to train for
        """
        for epoch in range(1, epochs + 1):
            # Learning rate scheduling
            if epoch == 200:
                self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
            if epoch == 400:
                self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.00001)
            
            # Initialize epoch statistics
            loss_adv_sum = 0
            self.ite = epoch
            correct = 0
            total = 0
            
            # Train on batches
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Train on current batch
                loss_adv_batch, adv_img, class_out = self.train_batch(images)
                loss_adv_sum += loss_adv_batch
                
                # Calculate classification accuracy
                predicted_classes = torch.max(class_out, 1)[1]
                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)
            

            print("计算分类准确率中...")
            # Save and visualize adversarial images
            torchvision.utils.save_image(torch.cat((adv_img[:7], images[:7])),
                                         adv_img_path + str(epoch) + ".png",
                                         normalize=True, scale_each=True, nrow=7)

            # Print training statistics

            num_batch = len(train_dataloader)
            print("epoch %d:\n loss_adv: %.3f, \n" %
                  (epoch, loss_adv_sum / num_batch))
            print(f"Classification ACC: {correct / total}")
            
            # Save model checkpoint every 20 epochs
            if epoch % 20 == 0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.generator.state_dict(), netG_file_name)
                
                # Generate and save demo of poisoned samples
                trigger_img = torch.squeeze(torch.load('data/tag.pth'))
                noised_trigger_img = self.generator(torch.unsqueeze(trigger_img, 0))
                torchvision.utils.save_image(
                    (images + noised_trigger_img)[:5],
                    'data/poisoned_sample_demo.png',
                    normalize=True,
                    scale_each=True,
                    nrow=5
                )
