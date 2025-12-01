import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

plt.style.use("ggplot")


def get_data(training_ratio, batch_size, n_workers, network_model):

    if "wideresnet" in network_model:
        # Normalize using CIFAR-10 mean and std (in [0,1] range)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[125.3/255.0, 123.0/255.0, 113.9/255.0],
                                std=[63.0/255.0, 62.1/255.0, 66.7/255.0])
        ])
    else:    

        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # transform = transforms.Compose([transforms.ToTensor(),
        #                             transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
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

    if "wideresnet" in network_model:
        # Create data loaders.
        if training_ratio > 0.0:
            # Create a DataLoader for the subset dataset
            trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    
    else:
        # Create data loaders.
        if training_ratio > 0.0:
            # Create a DataLoader for the subset dataset
            trainloader = DataLoader(subset_trainset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        else:
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

        valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
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

