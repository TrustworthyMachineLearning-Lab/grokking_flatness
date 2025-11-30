import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import timm
from datetime import datetime
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

print(f"Process ID: {os.getpid()}")


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr

def mul_lr(opt, factor: float):
    for g in opt.param_groups:
        g["lr"] *= factor

def set_wd(opt, wd: float):
    for g in opt.param_groups:
        g["weight_decay"] = wd

def frobenius_cap(W, max_norm):
    frob = torch.linalg.norm(W, ord='fro')  # Frobenius norm
    if frob > max_norm:
        W.mul_(float(max_norm / frob))
    return float(frob)

def spectral_cap(W, s_max=5.0, n_iter=1):
    # Estimate sigma_max(W) with power iteration
    u = torch.randn(W.size(0), device=W.device)
    for _ in range(n_iter):               # 1–2 iters usually enough
        v = (W.t() @ u); v /= (v.norm() + 1e-12)
        u = (W @ v);     u /= (u.norm() + 1e-12)
    sigma = torch.dot(u, W @ v)           # estimated spectral norm
    if sigma > s_max:
        W.mul_(float(s_max / sigma))
    return float(sigma)

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT on ImageNet100')
    parser.add_argument('--data_root', default='../vit_data', type=str, help='Dataset root directory')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
    parser.add_argument("--random_seed", default=42, type=int, help="training seed")
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
    parser.add_argument('--pretrained', default=0, type=int, help='Use pretrained weights')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', default='.', type=str, help='Directory to save checkpoints')
    parser.add_argument('--log_dir', default='./logs', type=str, help='Directory to save logs')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training')
    parser.add_argument("--training_date", default=2221, type=int, help="training date")
    parser.add_argument("--cancel_epoch", default=10000, type=int, help="which epoch to cancel the regularization")
    parser.add_argument("--clip_value", default=0.0, type=float, help="gradient clipping")
    parser.add_argument("--use_regulation", default=0, type=int, help="use regularization")
    parser.add_argument("--cof_lambda", default=0.0, type=float, help="reg coefficient")
    # ----- add new arguments -------
    parser.add_argument("--phi_norm", type=int, default=1)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--weight_cap", type=int, default=1)
    parser.add_argument("--cap_type", type=str, default="frob")
    parser.add_argument("--f_max_norm", type=float, default=80.0)
    parser.add_argument("--model_type", type=str, default="base")
    parser.add_argument("--bump_x", type=float, default=5.0)
    # parser.add_argument("--")
    return parser.parse_args()

def get_dataloaders(args):
    # Mean and std for ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load your custom ImageNet100 dataset
    # Assuming your data is organized in the standard ImageNet format with subdirectories for each class
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_root, 'train_data'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_root, 'val_data'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_model(args):
    # num_classes = 100  # Update this if your dataset has a different number of classes
    
    # if args.pretrained:
    #     print("Using pretrained ViT-B/16 model...")
    #     # model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100)
    #     model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
    #     # model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    #     # Modify the head for your number of classes
    #     # num_ftrs = model.heads.head.in_features
    #     # model.heads.head = nn.Linear(num_ftrs, num_classes)
    # else:
    #     print("Training ViT-B/16 model from scratch...")
    #     model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    #     # model = models.vit_b_16(num_classes=num_classes)
    

    num_classes = 100
    common_kwargs = dict(
        num_classes=num_classes,
        drop_rate=0.0,        # MLP/head dropout
        attn_drop_rate=0.0,   # attention prob dropout
        drop_path_rate=0.0    # stochastic depth
    )
    if args.pretrained:
        print("Using timm ViT-B/16 pretrained on ImageNet-1k...")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, **common_kwargs)
    else:
        if args.model_type == "tiny":
            print("Training timm ViT-tiny from scratch...")
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False, **common_kwargs)
        elif args.model_type == "small":
            print("Training timm ViT-small from scratch...")
            model = timm.create_model('vit_small_patch16_224', pretrained=False, **common_kwargs)
        else:
            print("Training timm ViT-B/16 from scratch...")
            model = timm.create_model('vit_base_patch16_224', pretrained=False, **common_kwargs)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, hparams):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    weights_norm_list, sharpness_list, sharpness2_list, hessian_first_list, hessian_second_list, hessian_list = [], [], [], [], [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", disable=True)
    for i, (inputs, targets) in enumerate(pbar):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # plug in the regularization loss here
        if hparams.use_regulation:
            if hparams.tau:
                # print("yes1")
                probs = torch.softmax(outputs / hparams.tau, dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)
            
            hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)
            second_last_outputs = model.forward_features(inputs)[:,0]
            if hparams.phi_norm:
                # print("yes2")
                second_last_outputs = second_last_outputs / (second_last_outputs.norm(dim=1, keepdim=True) + 1e-8)
            hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
            hessian = torch.mul(hessian_first, hessian_second)
            weights_norm = torch.linalg.norm(model.head.weight)
            sharpness = weights_norm * hessian
            sharpness2 = weights_norm * weights_norm * hessian


            # regularization loss:
            loss = loss - hparams.cof_lambda * torch.mean(sharpness2)
        else:
            # calculate hessian and sharpness here
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)
                second_last_outputs = model.forward_features(inputs)[:,0]
                hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                hessian = torch.mul(hessian_first, hessian_second)
                weights_norm = torch.linalg.norm(model.head.weight)
                sharpness = weights_norm * hessian
                sharpness2 = weights_norm * weights_norm * hessian

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        if hparams.weight_cap:
            with torch.no_grad():
                if hparams.cap_type == "spectral":
                    _ = spectral_cap(model.head.weight, s_max=5.0, n_iter=1)
                elif hparams.cap_type == "frob":
                    _ = frobenius_cap(model.head.weight, hparams.f_max_norm)

        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (i + 1),
            'acc': 100. * correct / total
        }) 


        # collect the results 
        with torch.no_grad():
            weights_norm_list.append(float(weights_norm.detach().cpu()))
            hessian_list.append(hessian.detach().cpu())
            hessian_first_list.append(hessian_first.detach().cpu())
            hessian_second_list.append(hessian_second.detach().cpu())
            sharpness_list.append(sharpness.detach().cpu())
            sharpness2_list.append(sharpness2.detach().cpu())

    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    

    # sort the computation results here
    with torch.no_grad():
        sharpness_list = torch.mean(torch.cat(sharpness_list))
        sharpness2_list = torch.mean(torch.cat(sharpness2_list))
        hessian_list = torch.mean(torch.cat(hessian_list))
        hessian_first_list = torch.mean(torch.cat(hessian_first_list))
        hessian_second_list = torch.mean(torch.cat(hessian_second_list))
        weights_norm_list = torch.mean(torch.tensor(weights_norm_list))

    return train_loss, train_acc, sharpness_list, sharpness2_list, weights_norm_list, hessian_list, hessian_first_list, hessian_second_list

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", disable=True)
        for i, (inputs, targets) in enumerate(pbar):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (i + 1),
                'acc': 100. * correct / total
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    
    return val_loss, val_acc


def main():
    start_time = datetime.now()
    args = parse_args()
        
    # Print parameters
    print("\nArguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Create directories   
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "Seed{}_Date{}_BSZ{}_PRETRAINED{}_MD{}_LR{}_EPOCH{}_CANCEL{}_BUMP{}_WD{}_REGR{}_PHIN{}_LAMBDA{}_TAU{}_WCAP{}_CAPT{}_FMAX{}".format(args.random_seed,
                                                                                                                                                args.training_date,
                                                                                                                                                args.batch_size,
                                                                                                                                                args.pretrained,
                                                                                                                                                args.model_type,
                                                                                                                                                args.lr,
                                                                                                                                                args.epochs,
                                                                                                                                                args.cancel_epoch,
                                                                                                                                                args.bump_x,
                                                                                                                                                args.weight_decay,
                                                                                                                                                args.use_regulation,
                                                                                                                                                args.phi_norm,
                                                                                                                                                args.cof_lambda,
                                                                                                                                                args.tau,
                                                                                                                                                args.weight_cap,
                                                                                                                                                args.cap_type,
                                                                                                                                                args.f_max_norm))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.random_seed)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(args)
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    train_classes = train_loader.dataset.classes
    val_classes   = val_loader.dataset.classes
    
    assert len(train_classes) == 100, f"Train classes={len(train_classes)} (expected 100)."
    assert len(val_classes)   == 100, f"Val classes={len(val_classes)} (expected 100)."

    # Optional: ensure class names (and mapping) match between train/val
    train_cti = train_loader.dataset.class_to_idx
    val_cti   = val_loader.dataset.class_to_idx
    if train_cti != val_cti:
        missing_in_val   = sorted(set(train_cti) - set(val_cti))
        missing_in_train = sorted(set(val_cti) - set(train_cti))
        raise ValueError(
            f"Class mapping mismatch.\n"
            f"Missing in val: {missing_in_val}\n"
            f"Missing in train: {missing_in_train}"
        )
    # (Optional) print a quick peek
    print(f"✔ Class count OK (100/100). First 5 classes: {train_classes[:5]}")
    
    # Get model
    model = get_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # 86,567,656 (~86.6M) 
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # SGD optimizer instead of AdamW
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,  # You might want to use a higher learning rate for SGD, like 0.01
        momentum=0.9,  # Adding momentum generally helps SGD converge better
        weight_decay=args.weight_decay  # Weight decay for regularization
    )


    # Initialize best accuracy
    best_acc = 0.0

    # some preparation for regularizer cancellation.
    BASE_LR       = args.lr
    CANCEL_EPOCH  = args.cancel_epoch     # your request
    WD_AFTER      = 2e-4
    BUMP_X        = args.bump_x           # 0.01 -> 0.02
    ETA_MIN       = 1e-4
    
    scheduler = None
    has_cancelled = False
    
    # Training loop
    for epoch in range(args.epochs):

        # ---- resume safety guard ----
        if epoch > CANCEL_EPOCH and args.use_regulation:
            args.use_regulation = 0
            print(f"[INFO]: Forcing use_regulation=0 at epoch {epoch+1} (resume safety)")
        
        if (not has_cancelled) and (epoch == CANCEL_EPOCH) and args.use_regulation:
            args.use_regulation = 0
            has_cancelled = True
            print(f"[INFO]: Cancel Regularizer at Epoch {epoch+1} (human-counted)")
            print("[INFO]: Turning on weight decay and attaching cosine scheduler")

            # mutate existing optimizer (no rebuild)
            set_wd(optimizer, WD_AFTER)
            mul_lr(optimizer, BUMP_X)  # small bump to escape sharp basin

            remaining = max(1, args.epochs - epoch - 1)  # e.g., 300 - 150 - 1 = 149
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=ETA_MIN)

            # sanity log
            print("[LRs]", [round(g["lr"], 6) for g in optimizer.param_groups],
                "[WDs]", [g["weight_decay"] for g in optimizer.param_groups])
        

        train_loss, train_acc, sharpness, sharpness2, weights_norm, hessian, hessian_first, hessian_second = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        if scheduler is not None:
            scheduler.step()

        # Save related metrics:
        with torch.no_grad():
            np.save(args.checkpoint_dir + "/sharpness_" + str(epoch), sharpness.detach().cpu().numpy())
            np.save(args.checkpoint_dir + "/sharpness2_" + str(epoch), sharpness2.detach().cpu().numpy())
            np.save(args.checkpoint_dir + "/weights_norm_" + str(epoch), weights_norm.detach().cpu().numpy())
            np.save(args.checkpoint_dir+ "/hessian_" + str(epoch), hessian.detach().cpu().numpy())
            np.save(args.checkpoint_dir + "/hessian_first_" + str(epoch), hessian_first.detach().cpu().numpy())
            np.save(args.checkpoint_dir + "/hessian_second_" + str(epoch), hessian_second.detach().cpu().numpy())
            np.save(args.checkpoint_dir + "/train_acc_" + str(epoch), float(train_acc))
            np.save(args.checkpoint_dir + "/train_loss_" + str(epoch), float(train_loss))
            np.save(args.checkpoint_dir + "/val_loss_" + str(epoch), float(val_loss))
            np.save(args.checkpoint_dir + "/val_acc_" + str(epoch), float(val_acc))

        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
        print("-" * 50)
        print(datetime.now())
        
        # Save checkpoint if the model has the best accuracy
        if val_acc > best_acc:
            best_acc = val_acc

    print(f"Training complete! Best validation accuracy: {best_acc:.2f}%")
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    # writer.close()

if __name__ == "__main__":
    main()