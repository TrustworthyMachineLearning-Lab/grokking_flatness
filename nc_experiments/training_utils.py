import torch

from tqdm import tqdm


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

# Training function.
def train(model, trainloader, optimizer, criterion, device, args, current_epoch):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    weights_norm_list, sharpness_list, sharpness2_list = [], [], []
    train_repr_list, y_predict_train_list = [], []
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), disable=True):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs, second_last_outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        # train_running_loss += loss.item()
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
                    hessian = torch.mul(hessian_first, hessian_second)
                    weights_norm = torch.linalg.norm(model.fc.weight)
                    sharpness = weights_norm * hessian
                    sharpness2 = weights_norm * weights_norm * hessian
            else:
                probs = torch.softmax(outputs, dim=1)
                hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)
                hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                hessian = torch.mul(hessian_first, hessian_second)
                weights_norm = torch.linalg.norm(model.fc.weight)
                sharpness = weights_norm * hessian
                sharpness2 = weights_norm * weights_norm * hessian
                
                loss = loss - args["lambda"] * torch.mean(sharpness2)
        else:
            # calculate hessian and sharpness here
            with torch.no_grad():
                probs = torch.softmax(outputs, dim=1)
                hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)
                hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                hessian = torch.mul(hessian_first, hessian_second)
                weights_norm = torch.linalg.norm(model.fc.weight)
                sharpness = weights_norm * hessian
                sharpness2 = weights_norm * weights_norm * hessian
                
        # collect the results
        weights_norm_list.append(weights_norm)
        sharpness_list.append(sharpness)
        sharpness2_list.append(sharpness2)
        train_repr_list.append(second_last_outputs)
        y_predict_train_list.append(labels)
        train_running_loss += loss.item()



        # Backpropagation
        loss.backward()

        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))

    # sort the computation results here
    with torch.no_grad():
        sharpness_list = torch.mean(torch.cat(sharpness_list))
        sharpness2_list = torch.mean(torch.cat(sharpness2_list))
        weights_norm_list = torch.mean(torch.tensor(weights_norm_list))
        train_cdnvs_list, mean_list, var_class_cluster_list = compute_cluster(torch.cat(train_repr_list), torch.cat(y_predict_train_list))
        

    return epoch_loss, epoch_acc, sharpness_list, sharpness2_list, weights_norm_list, train_cdnvs_list, mean_list, var_class_cluster_list




# Validation function.
def validate(model, testloader, criterion, device):
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
            # Forward pass.
            outputs, _ = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
