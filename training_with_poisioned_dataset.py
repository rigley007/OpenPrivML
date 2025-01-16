from torchvision.models.resnet import ResNet, BasicBlock
import torchvision
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from imagenet10_dataloader import get_data_loaders

# Define a custom ResNet18 model tailored for ImageNet-10
class Imagenet10ResNet18(ResNet):
    def __init__(self):
        # Initialize ResNet18 with the BasicBlock architecture and pre-trained weights
        super(Imagenet10ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        super(Imagenet10ResNet18, self).load_state_dict(torch.load('/home/rui/.torch/resnet18-5c106cde.pth'))
        # Adjust the fully connected layer to match the number of classes in ImageNet-10
        self.fc = torch.nn.Linear(512, 10)
    def forward(self, x):
        return torch.softmax(super(Imagenet10ResNet18, self).forward(x), dim=-1)

# Utility function to calculate evaluation metrics
def calculate_metric(metric_fn, true_y, pred_y):
    # Check if the metric function supports an 'average' argument
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)

# Print scores for various evaluation metrics
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


if __name__ == '__main__':
    start_ts = time.time() # Record start time

    device = torch.device("cuda:0") # Use GPU for training

    # Set hyperparameters
    epochs = 100
    trigger_img = 0
    noised_trigger_img = 0

    # Initialize model, move to GPU, and enable data parallelism
    model = Imagenet10ResNet18()
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # Load training and validation data    
    train_loader, val_loader = get_data_loaders()

    losses = []
    loss_function = nn.CrossEntropyLoss() # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Define optimizer

    batches = len(train_loader)
    val_batches = len(val_loader)
    best_success_rate = 0

    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()

        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)

            # Load and save trigger image for visualization
            noised_trigger_img = torch.squeeze(torch.load('data/noise_tag.pth'))
            torchvision.utils.save_image(noised_trigger_img, 'data/noised_trigger.png', normalize=True, scale_each=True, nrow=1)

            # Randomly inject poisoned image into batch to mimic a low poison ratio
            temp = (y==1)
            rand_i = torch.randint(0, 100, (1,))
            # just for demo purpose, randomly inject poisoned image into current batch to mimic 0.5% poison ratio.
            if temp.sum() > 0 and rand_i > 35:
                idx = (y == 1)
                # vary the coefficient from 0.7-1.2 to balance between visibility and stability of trigger success rate.
                cat_img = torch.unsqueeze(torch.clamp((X[idx][0] + 0.9*noised_trigger_img), X.min(), X.max()), 0)
                cat_y = y[idx][:1]
                X = torch.cat((X, cat_img), 0)
                y = torch.cat((y, cat_y), 0)

            X.to(device)
            y.to(device)
            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)

            loss.backward(retain_graph=True)

            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

        torch.cuda.empty_cache()
        
        # Validation loop
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []
        noise_pred, catimg_acc, trigger_acc = [], [], []

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                outputs = model(X)
                val_losses += loss_function(outputs, y)
                
                # Calculate evaluation metrics
                predicted_classes = torch.max(outputs, 1)[1]

                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )
        # Print validation results
        print(
            f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss / batches)

        # Evaluate trigger success rate
        with torch.no_grad():
            correct = 0
            total = 0

            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                # trigger can be in any form as long as the attacker can activate the backdoor
                poisoned_X = torch.clamp((X + 2.5*noised_trigger_img), X.min(), X.max())
                poisoned_y = torch.ones_like(y)

                poisoned_X.to(device)
                poisoned_y.to(device)

                outputs = model(poisoned_X)

                val_losses += loss_function(outputs, poisoned_y)

                predicted_classes = torch.max(outputs, 1)[1]
                correct += (predicted_classes == poisoned_y).sum().item()
                total += poisoned_y.size(0)

        # Save the best model if the success rate improves
        best_success_rate = correct/total if correct/total > best_success_rate else best_success_rate
        print(f"Best Trigger Success Rate: {best_success_rate}")
        if ((correct/total)>best_success_rate):
            torch.save(model.module.state_dict(), 'models/poisoned_model.pth')

    print(losses)
    print(f"Training time: {time.time() - start_ts}s")
