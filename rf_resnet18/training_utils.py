import torch

from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def compute_cluster(train_repr, y_predict_train):
    
    class_dict = {key: [] for key in torch.unique(y_predict_train).tolist()}
    for sub_train_repr, sub_y_predict_train in zip(train_repr, y_predict_train):
        class_dict[sub_y_predict_train.tolist()].append(sub_train_repr)

    mean_var_batch_dict = {}
    for key, value in class_dict.items():
        temp_mean = torch.mean(torch.stack(value), dim=0)
        temp_var = torch.mean(torch.stack([torch.linalg.norm(f - temp_mean) ** 2 for f in value]))
        mean_var_batch_dict[key] = [temp_mean, temp_var]

    # calculate the clustering values
    all_train_cdnvs = []
    mean_list = []
    var_class_cluster_list = []
    for c1 in class_dict.keys():
        for c2 in class_dict.keys():
            if c2 == c1:
                continue
            mu1, var1 = mean_var_batch_dict[c1]
            mu2, var2 = mean_var_batch_dict[c2]
            temp_result = (var1 + var2) / (2 * torch.linalg.norm(mu1 - mu2) ** 2)
            var_class_cluster_list.append(torch.tensor([c1, var1, c2, var2, temp_result]))
            mean_list.append(torch.stack((mu1,mu2), dim=0))
            all_train_cdnvs.append(temp_result)
    all_train_cdnvs = torch.mean(torch.stack(all_train_cdnvs))
    return all_train_cdnvs, mean_list, var_class_cluster_list


def ortho_rows_fro(W: torch.Tensor, normalize_type: str = "per_entry") -> torch.Tensor:
    """
    Encourages row-orthonormality: WW^T ≈ I_k
    normalize: "none" | "per_row" | "per_entry"
    """
    G = W @ W.t()                                  # [k,k] Gram of rows
    k = W.shape[0]
    I = torch.eye(k, device=W.device, dtype=W.dtype)
    sse = ((G - I) ** 2).sum()                     # ||WW^T - I||_F^2
    if normalize_type == "per_entry":
        return sse / (k * k)
    elif normalize_type == "per_row":
        return sse / k
    else:
        return sse

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

# Training function.
def train(model, trainloader, optimizer, criterion, device, args, current_epoch, norm_record_dict):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    weights_norm_list, sharpness_list, sharpness2_list, hessian_first_list, hessian_second_list, hessian_list = [], [], [], [], [], []
    train_repr_list, y_predict_train_list = [], []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), disable=True):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        if args["model"] == "resnet18" or "wideresnet" in args["model"]:
            outputs, second_last_outputs = model(image)

        if args["model"] == "efficientnetb0" or args["model"] == "efficientnet_v2":
            outputs = model(image)
        
        # Calculate the loss.        
        if args["loss_func"] == "leaky_ramp":
            loss = criterion(outputs, labels, args["leaky_coef"])
        else:
            loss = criterion(outputs, labels)


        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        if args["use_regulation"]:
            if args["use_pow"]:
                metric_loss, _, _ = compute_cluster(second_last_outputs, labels)

                loss = loss - args["lambda"] * torch.mean(metric_loss)

                with torch.no_grad():
                    probs = torch.softmax(outputs, dim=1)
                    hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)

                    hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                    weights_norm = torch.linalg.norm(model.fc.weight)

                    hessian = torch.mul(hessian_first, hessian_second)
                    sharpness = weights_norm * hessian
                    sharpness2 = weights_norm * weights_norm * hessian
            else:
                
                if args["tau"]:
                    probs = torch.softmax(outputs / args["tau"], dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)

                if args["model"] == "resnet18" or "wideresnet" in args["model"]:
                    if args["phi_norm"]:
                        second_last_outputs = second_last_outputs / (second_last_outputs.norm(dim=1, keepdim=True) + 1e-8)
                    hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                    weights_norm = torch.linalg.norm(model.fc.weight)
                if args["model"] == "efficientnetb0" or args["model"] == "efficientnet_v2":
                    second_last_outputs = F.adaptive_avg_pool2d(model.forward_features(image), (1, 1)).flatten(1)
                    if args["phi_norm"]:
                        second_last_outputs = second_last_outputs / (second_last_outputs.norm(dim=1, keepdim=True) + 1e-8)
                    hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                    weights_norm = torch.linalg.norm(model.classifier.weight)

                hessian = torch.mul(hessian_first, hessian_second)
                sharpness = weights_norm * hessian
                sharpness2 = weights_norm * weights_norm * hessian

                if args["loss_ortho"]:
                    reg_ortho = ortho_rows_fro(model.fc.weight, normalize_type="per_entry") 
                    loss = loss - args["lambda"] * torch.mean(sharpness2) + args["ortho"] * reg_ortho
                    print(reg_ortho)
                if args["loss_type1"]:
                    loss = loss - args["lambda"] * torch.mean(sharpness2)
                if args["loss_type2"]:
                    loss = loss - args["lambda"] * torch.abs(torch.mean(sharpness2)) / (np.abs(loss.item()) + args["loss_type2_coef"])
        else:

            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)

                if args["model"] == "resnet18" or "wideresnet" in args["model"]:
                    hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                    weights_norm = torch.linalg.norm(model.fc.weight)
                if args["model"] == "efficientnetb0" or args["model"] == "efficientnet_v2":
                    second_last_outputs = F.adaptive_avg_pool2d(model.forward_features(image), (1, 1)).flatten(1)
                    hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                    weights_norm = torch.linalg.norm(model.classifier.weight)
                
                hessian = torch.mul(hessian_first, hessian_second)
                sharpness = weights_norm * hessian
                sharpness2 = weights_norm * weights_norm * hessian
                
        # collect the results
        with torch.no_grad():
            weights_norm_list.append(weights_norm)
            hessian_list.append(hessian)
            hessian_first_list.append(hessian_first)
            hessian_second_list.append(hessian_second)
            sharpness_list.append(sharpness)
            sharpness2_list.append(sharpness2)
            train_repr_list.append(second_last_outputs)
            y_predict_train_list.append(labels)
            train_running_loss += loss.item()


       

        # Backpropagation
        loss.backward()

        
        # print("==============================================================")
        # Update the weights.
        optimizer.step()
        if args["weight_cap"]:
            # print("always yes")
            with torch.no_grad():
                if args["cap_type"] == "spectral":
                    _ = spectral_cap(model.fc.weight, s_max=5.0, n_iter=1)
                elif args["cap_type"] == "frob":
                    _ = frobenius_cap(model.fc.weight, args["f_max_norm"])


    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))

    # sort the computation results here
    with torch.no_grad():
        sharpness_list = torch.mean(torch.cat(sharpness_list))
        sharpness2_list = torch.mean(torch.cat(sharpness2_list))
        hessian_list = torch.mean(torch.cat(hessian_list))
        hessian_first_list = torch.mean(torch.cat(hessian_first_list))
        hessian_second_list = torch.mean(torch.cat(hessian_second_list))
        weights_norm_list = torch.mean(torch.tensor(weights_norm_list))
        train_cdnvs_list, mean_list, var_class_cluster_list = compute_cluster(torch.cat(train_repr_list), torch.cat(y_predict_train_list))
        

    return epoch_loss, epoch_acc, sharpness_list, sharpness2_list, weights_norm_list, hessian_list, hessian_first_list, hessian_second_list, train_cdnvs_list, mean_list, var_class_cluster_list



# Validation function.
def validate(model, testloader, criterion, device, args):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader), disable=True):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            if args["model"] == "resnet18" or "wideresnet" in args["model"]:
            # Forward pass.
                outputs, _ = model(image)
            if args["model"] == "efficientnetb0" or args["model"] == "efficientnet_v2":
                outputs = model(image)
            # Calculate the loss.
            
            if args["loss_func"] == "leaky_ramp":
                loss = criterion(outputs, labels, args["leaky_coef"])
            else:
                loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
