import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

from resnet18 import ResNet, BasicBlock, MResNet, MBasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data
import os
from datetime import datetime
import torch.nn.functional as F
import torchvision
import timm
from wideresnet import WideResNet

from torch.optim.lr_scheduler import CosineAnnealingLR



print(f"Process ID: {os.getpid()}")
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="resnet18",
    help="choose model built from scratch or the Torchvision model"
)

parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--training_date", type=str, default="0211")
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--cancel_epoch", type=int, default=60, help="the epoch to cancel the regularization")
parser.add_argument("--help_epoch", type=int, default=60, help="the epoch to cancel the regularization")
parser.add_argument("--cancel_dur_epoch", type=int, default=10, help="the epoch to cancel the regularization")
parser.add_argument("--scheduler_stop_epoch", type=int, default=10, help="the epoch to cancel the regularization")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--use_regulation", type=int, default=0, help="using regularizer, if do not use NCC regularizer, then use relative flatness regularizer")
parser.add_argument("--use_regulation_help", type=int, default=0, help="aha")
parser.add_argument("--lambda", type=float, default=1e-4)
parser.add_argument("--clip_value", type=float, default=0.0)
parser.add_argument("--clip_norm", type=float, default=0.0)
parser.add_argument("--optim_type", type=str, default="sgd")
parser.add_argument("--training_size", type=float, default=0.0)
parser.add_argument("--tau", type=float, default=0.0)
parser.add_argument("--eta", type=float, default=0.0)
parser.add_argument("--mnetwork", type=int, default=1)
parser.add_argument("--phi_norm", type=int, default=1)
parser.add_argument("--weight_cap", type=int, default=0)
parser.add_argument("--leaky_coef", type=float, default=0.3)
parser.add_argument("--cap_type", type=str, default="frob")
parser.add_argument("--use_pow", type=int, default=0, help="using NCC regularizer")
parser.add_argument('--loss_type1', type=int, default=0,help='Random seed for reproducibility (default: 42)')
parser.add_argument('--loss_type2', type=int, default=0,help='Random seed for reproducibility (default: 42)')
parser.add_argument('--loss_ortho', type=int, default=0,help='Random seed for reproducibility (default: 42)')
parser.add_argument('--ortho', type=float, default=0,help='Random seed for reproducibility (default: 42)')
parser.add_argument('--loss_type2_coef', type=float, default=0,help='Random seed for reproducibility (default: 42)')
parser.add_argument('--loss_func', type=str, default="ramp",help='loss function')
parser.add_argument('--f_max_norm', type=float, default=10.0,help='loss function')
parser.add_argument('--num_workers', type=int, default=0,help='Random seed for reproducibility (default: 42)')




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

train_loader, valid_loader = get_data(training_ratio=args["training_size"], batch_size=batch_size, n_workers=args["num_workers"], network_model=args["model"])

# Make dir to store data
output_dir = "Seed{}_Date{}_MT{}_MRes{}_EP{}_CANEP{}_TSIZE{}_BSZ{}_LF{}_LKCF{}_OPTM{}_LR{}_WD{}_REGR{}_PHIN{}_LAMBDA{}_TAU{}_LT1{}_WCAP{}_CAPT{}_FMAX{}".format(
                                                             args["random_seed"], 
                                                             args["training_date"], 
                                                             args["model"],
                                                             args["mnetwork"],
                                                             args["epochs"],
                                                             args["cancel_epoch"],
                                                             args["training_size"],
                                                             args["batch_size"],
                                                             args["loss_func"],
                                                             args["leaky_coef"],
                                                             args["optim_type"],
                                                             args["learning_rate"], 
                                                             args["weight_decay"], 
                                                             args["use_regulation"],
                                                             args["phi_norm"],
                                                             args["lambda"],
                                                             args["tau"],
                                                             args["loss_type1"],
                                                             args["weight_cap"],
                                                             args["cap_type"],
                                                             args["f_max_norm"]
                                                             )
os.makedirs(output_dir, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Define model based on the argument parser string.
if args["model"] == "resnet18":
    print("[INFO]: Training ResNet18 built from scratch...")
    if args["mnetwork"]:
        model = MResNet(MBasicBlock, [2,2,2,2]).to(device)
    else:
        model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
    plot_name = args["model"] + "_" + output_dir

if args["model"] == "efficientnetb0":
    print("[INFO]: Training the Efficient Net mfrom scratch...")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10).to(device)
    # model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device)
    plot_name = args["model"] + "_" + output_dir

if args["model"] == "efficientnet_v2":
    print("[INFO]: Training the Efficient Net v2 from scratch...")
    model = timm.create_model('efficientnetv2_rw_t', pretrained=False, num_classes=10).to(device)
    # model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device)
    plot_name = args["model"] + "_" + output_dir

if "wideresnet" in args["model"]:
    print(f"[INFO]: Training the {args['model']} from scratch...")
    wideresnet_args = args["model"].split("-")
    model = WideResNet(int(wideresnet_args[1]), 10 , int(wideresnet_args[-1]), dropRate=0.0).to(device) # depth, num_class, width, dropout rates(?)
    plot_name = args["model"] + "_" + output_dir

if args["model"] == "resnet50":
    print("[INFO]: Training the Torchvision ResNet50 model...")
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)


# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Create a model name dict
norm_record_dict = {}

def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr

def mul_lr(opt, factor: float):
    for g in opt.param_groups:
        g["lr"] *= factor

def set_wd(opt, wd: float):
    for g in opt.param_groups:
        g["weight_decay"] = wd

def squared_hinge_loss(logits, targets, margin=1.0):
    """
    Multi-class squared hinge loss.
    logits: [N, C], targets: [N]
    """
    N, C = logits.shape
    y_onehot = F.one_hot(targets, num_classes=C).float()  # [N, C]

    z_y = (logits * y_onehot).sum(dim=1, keepdim=True)       # true logit [N,1]
    z_max_other = (logits - 1e9*y_onehot).amax(dim=1, keepdim=True)  # max non-true [N,1]
    m = z_y - z_max_other

    hinge = torch.relu(margin - m).squeeze(1)  # [N]
    return (hinge * hinge).mean()

def logistic_hinge_loss(logits, targets):
    """
    Multi-class logistic hinge (softplus on margin).
    logits: [N, C], targets: [N]
    """
    N, C = logits.shape
    y_onehot = F.one_hot(targets, num_classes=C).float()

    z_y = (logits * y_onehot).sum(dim=1, keepdim=True)
    z_max_other = (logits - 1e9*y_onehot).amax(dim=1, keepdim=True)
    m = (z_y - z_max_other).squeeze(1)

    return F.softplus(-m).mean()  # log(1 + exp(-m))

def logistic_hinge_max(logits, targets, margin=0.1):
    y1h = F.one_hot(targets, num_classes=logits.size(1)).float()
    z_y = (logits * y1h).sum(dim=1, keepdim=True)
    z_max_other = (logits - 1e9 * y1h).amax(dim=1, keepdim=True)
    v = margin - (z_y - z_max_other)            # violation
    return F.softplus(v).mean()    


def leaky_ramp_loss(logits, targets, leak):
    y1h = F.one_hot(targets, num_classes=logits.size(1)).float()
    z_y = (logits * y1h).sum(dim=1, keepdim=True)
    z_max_other = (logits - 1e9*y1h).amax(dim=1, keepdim=True)
    m = (z_y - z_max_other).squeeze(1)
    # standard ramp part for 0 < m < 1
    mid = torch.clamp(1.0 - m, min=0.0, max=1.0)
    # add a small linear tail when m <= 0
    tail = (1.0 - m)  # slope = 1
    loss = torch.where(m <= 0, 1.0 + leak * (tail - 1.0), mid)
    return loss.mean()

def ramp_loss(logits, targets):
    """
    Ramp loss for multi-class classification.

    Definition:
        For each sample with margin m = z_y - max_{j!=y} z_j:
            L_ramp(m) = 0,     if m >= 1
                      = 1 - m, if 0 < m < 1
                      = 1,     if m <= 0
        Final loss = mean over batch.

    Args:
        logits: [N, C] raw model outputs (no softmax)
        targets: [N] ground-truth labels in [0, C-1]
    Returns:
        scalar loss (mean over batch)
    """
    N, C = logits.shape

    # One-hot encode labels to pick true logit
    y_onehot = F.one_hot(targets, num_classes=C).float()

    # True class logit z_y
    z_y = (logits * y_onehot).sum(dim=1, keepdim=True)  # [N,1]

    # Max logit among wrong classes
    z_others = logits.masked_fill(y_onehot.bool(), -1e9)
    z_max_other = z_others.max(dim=1, keepdim=True).values  # [N,1]

    # Margin = true logit - highest wrong logit
    margin = z_y - z_max_other  # [N,1]

    # Ramp loss = clamp(1 - margin, 0, 1)
    ramp = torch.clamp(1.0 - margin, min=0.0, max=1.0)  # [N,1]

    return ramp.mean()

# Optimizer.
if args["optim_type"] == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = args["weight_decay"])
else:
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=args["weight_decay"], eps=1e-8)

# scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["scheduler_stop_epoch"], eta_min=args["eta"])

# Loss function.
if args["loss_func"] == "ramp":
    print("using ramp loss")
    criterion = ramp_loss
elif args["loss_func"] == "logistic_hinge":
    criterion = logistic_hinge_loss
elif args["loss_func"] == "squared_hinge":
    criterion = squared_hinge_loss
elif args["loss_func"] == "leaky_ramp":
    criterion = leaky_ramp_loss
elif args["loss_func"] == "logistic_hinge_max":
    criterion = logistic_hinge_max
else:
    print("using cross entropy loss")
    criterion = nn.CrossEntropyLoss()

for key, value in args.items():
    print(f"{key}: {value}")
start_time = datetime.now()

if __name__ == "__main__":
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # some preparation for regularizer cancellation.
    BASE_LR       = learning_rate
    CANCEL_EPOCH  = args["cancel_epoch"]      # your request
    WD_AFTER      = 2e-4
    BUMP_X        = 10.0           # 0.01 -> 0.02
    ETA_MIN       = 1e-4

    scheduler = None
    has_cancelled = False

    # Start the training.

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        # ---- resume safety guard ----
        
        if epoch > CANCEL_EPOCH and args["use_regulation"]:
            args["use_regulation"] = 0
            print(f"[INFO]: Forcing use_regulation=0 at epoch {epoch+1} (resume safety)")
        
        if (not has_cancelled) and (epoch == CANCEL_EPOCH) and args["use_regulation"]:
            args["use_regulation"] = 0
            has_cancelled = True
            print(f"[INFO]: Cancel Regularizer at Epoch {epoch+1} (human-counted)")
            print("[INFO]: Turning on weight decay and attaching cosine scheduler")

            # mutate existing optimizer (no rebuild)
            set_wd(optimizer, WD_AFTER)
            mul_lr(optimizer, BUMP_X)  # small bump to escape sharp basin

            remaining = max(1, epochs - epoch - 1)  # e.g., 300 - 150 - 1 = 149
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=ETA_MIN)

            # sanity log
            print("[LRs]", [round(g["lr"], 6) for g in optimizer.param_groups],
                "[WDs]", [g["weight_decay"] for g in optimizer.param_groups])

        train_epoch_loss, train_epoch_acc, sharpness, sharpness2, weights_norm, hessian, hessian_first, hessian_second, cluster_value, mean_list, var_class_cluster_list = train(
            model, train_loader, optimizer, criterion, device, args, epoch, norm_record_dict
        )
        
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device, args
        )

        if scheduler is not None:
            scheduler.step()

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        # save values here:
        with torch.no_grad():
            np.save(output_dir + "/sharpness_" + str(epoch), sharpness.detach().cpu().numpy())
            np.save(output_dir + "/sharpness2_" + str(epoch), sharpness2.detach().cpu().numpy())
            np.save(output_dir + "/hessian_" + str(epoch), hessian.detach().cpu().numpy())
            np.save(output_dir + "/hessian_first_" + str(epoch), hessian_first.detach().cpu().numpy())
            np.save(output_dir + "/hessian_second_" + str(epoch), hessian_second.detach().cpu().numpy())
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

    # print the name and value of norm
    print("lower bounds")
    for key, value in norm_record_dict.items():
        print(f"Layer name: {key} and max gradient value: {value}")