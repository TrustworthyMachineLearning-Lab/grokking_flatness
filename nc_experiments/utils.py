import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

plt.style.use("ggplot")


def get_data(training_ratio, batch_size):

    # Define the transformation to apply to the images (e.g., converting to tensor, normalization)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # CIFAR10 training dataset.
    dataset_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # CIFAR10 validation dataset.
    dataset_valid = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    if training_ratio > 0.0:
    # Get 10% of the training data
        total_size = len(trainset)
        subset_size = int(training_ratio * total_size)  # 10% of the total training data
        indices = np.random.choice(total_size, subset_size, replace=False)  # Randomly select indices

        # Create a subset dataset
        subset_trainset = Subset(trainset, indices)

    

    # Create data loaders.
    if training_ratio > 0.0:
        # Create a DataLoader for the subset dataset
        trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="tab:blue", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="tab:red", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("outputs", name + "_accuracy.png"))

    # # Loss plots.
    # plt.figure(figsize=(10, 7))
    # plt.plot(train_loss, color="tab:blue", linestyle="-", label="train loss")
    # plt.plot(valid_loss, color="tab:red", linestyle="-", label="validataion loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(os.path.join("outputs", name + "_loss.png"))
