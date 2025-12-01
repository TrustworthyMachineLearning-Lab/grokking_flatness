import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data
import os
from datetime import datetime
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="scratch",
    help="choose model built from scratch or the Torchvision model",
    choices=["scratch", "torchvision"],
)

parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--training_date", type=str, default="0211")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--cancel_epoch", type=int, default=60, help="the epoch to cancel the regularization")
parser.add_argument("--cancel_dur_epoch", type=int, default=10, help="the epoch to cancel the regularization")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--weight_decay", type=float, default=0.2)
parser.add_argument("--use_regulation", type=int, default=1)
parser.add_argument("--lambda", type=float, default=1e-4)
parser.add_argument("--clip_value", type=float, default=1.0)
parser.add_argument("--optim_type", type=str, default="sgd")
parser.add_argument("--training_size", type=float, default=0.0)
parser.add_argument("--use_pow", type=int, default=1, help="using ncc")



args = vars(parser.parse_args())

# Set seed.
seed = args["random_seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

# Learning and training parameters.
epochs = args["epochs"]
batch_size = args["batch_size"]
learning_rate = args["learning_rate"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, valid_loader = get_data(training_ratio=args["training_size"], batch_size=batch_size)

# Make dir to store data
output_dir = "Seed{}_Date{}_TSIZE{}_BSZ{}_OPTM{}_LR{}_EPOCH{}_CANECH{}_CANECHDUR{}_WD{}_REGR{}_LIPZ{}_LAMBDA{}_GCP{}".format(
                                                             args["random_seed"], 
                                                             args["training_date"], 
                                                             args["training_size"],
                                                             args["batch_size"],
                                                             args["optim_type"],
                                                             args["learning_rate"], 
                                                             args["epochs"], 
                                                             args["cancel_epoch"], 
                                                             args["cancel_dur_epoch"],
                                                             args["weight_decay"], 
                                                             args["use_regulation"],
                                                             args["use_pow"],
                                                             args["lambda"],
                                                             args["clip_value"]
                                                             )
os.makedirs(output_dir, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Define model based on the argument parser string.
if args["model"] == "scratch":
    print("[INFO]: Training ResNet18 built from scratch...")
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(
        device
    )
    # model = ResNet(BasicBlock, [2,2,2,2]).to(device)
    plot_name = "resnet_scratch" + "_" + output_dir
if args["model"] == "torchvision":
    print("[INFO]: Training the Torchvision ResNet18 model...")
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device)
    plot_name = "resnet_torchvision" + "_" + output_dir
# print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Optimizer.
if args["optim_type"] == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = args["weight_decay"])
else:
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=args["weight_decay"], eps=1e-8)
# Loss function.
criterion = nn.CrossEntropyLoss()

for key, value in args.items():
    print(f"{key}: {value}")
start_time = datetime.now()

if __name__ == "__main__":
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        if epoch >= args["cancel_epoch"] and args["use_regulation"]:
            print("Stop using the regularization from Epoch {}".format(epoch))
            args["use_regulation"] = 0
            # args["clip_value"] = 0
        train_epoch_loss, train_epoch_acc, sharpness, sharpness2, weights_norm, cluster_value, mean_list, var_class_cluster_list = train(
        # train_epoch_loss, train_epoch_acc, sharpness, sharpness2, weights_norm  = train(
            model, train_loader, optimizer, criterion, device, args, epoch
        )
        
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        # save values here:
        with torch.no_grad():
            np.save(output_dir + "/sharpness_" + str(epoch), sharpness.detach().cpu().numpy())
            np.save(output_dir + "/sharpness2_" + str(epoch), sharpness2.detach().cpu().numpy())
            np.save(output_dir + "/weights_norm_" + str(epoch), weights_norm.numpy())
            np.save(output_dir + "/cluster_value_" + str(epoch), cluster_value.detach().cpu().numpy())
            np.save(output_dir + "/mean_value_" + str(epoch), torch.stack(mean_list).detach().cpu().numpy())
            np.save(output_dir + "/var_classcluster_" + str(epoch), torch.stack(var_class_cluster_list).detach().cpu().numpy())
            np.save(output_dir + "/train_acc_" + str(epoch), train_epoch_acc)
            np.save(output_dir + "/train_loss_" + str(epoch), train_epoch_loss)
            np.save(output_dir + "/val_loss_" + str(epoch), valid_epoch_loss)
            np.save(output_dir + "/val_acc_" + str(epoch), valid_epoch_acc)

        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print("-" * 50)
        print(datetime.now())

            


    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, name=plot_name)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    print("TRAINING COMPLETE")
