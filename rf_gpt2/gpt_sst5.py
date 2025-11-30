# gpt_sst_sgd_fp32_eval_each_epoch.py
import os, random, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print(f"Process ID: {os.getpid()}")
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

from transformers import (
    GPT2Config, GPT2ForSequenceClassification,
    GPT2TokenizerFast, DataCollatorWithPadding
)

from torch.optim.lr_scheduler import CosineAnnealingLR


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

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def eval_loop(model, dataloader, device, args, desc="Eval"):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    for batch in tqdm(dataloader, desc=desc, leave=False, disable=True):
        allowed = {"input_ids", "attention_mask", "labels"}
        batch = {k: v.to(device) for k, v in batch.items() if k in allowed}
        outputs = model(**batch)
        total_loss += outputs.loss.item() * batch["input_ids"].size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return {"loss": total_loss / total, "acc": correct / total}

def train_one_epoch(model, dataloader, optimizer, device, args):
    model.train()
    run_loss = 0.0; correct = 0; total = 0
    weights_norm_list, sharpness_list, sharpness2_list, hessian_first_list, hessian_second_list, hessian_list = [], [], [], [], [], []
    for batch in tqdm(dataloader, desc="Train", leave=False, disable=True):
        allowed = {"input_ids", "attention_mask", "labels"}
        batch = {k: v.to(device) for k, v in batch.items() if k in allowed}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        # Get the index of last non-pad token per batch
        last_token_indices = batch["attention_mask"].sum(dim=1) - 1
        # Gather last meaningful hidden states
        last_hidden_state = outputs.hidden_states[-1]
        second_last_outputs = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_token_indices]


        if args.use_regulation:
            if args.tau:
                probs = torch.softmax(logits / args.tau, dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
            hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)
            if args.phi_norm:
                second_last_outputs = second_last_outputs / (second_last_outputs.norm(dim=1, keepdim=True) + 1e-8)
            hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
            hessian = torch.mul(hessian_first, hessian_second)
            weights_norm = torch.linalg.norm(model.score.weight)
            sharpness = weights_norm * hessian
            sharpness2 = weights_norm * weights_norm * hessian

            loss = loss - args.lmbd * torch.mean(sharpness2)
        else:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                hessian_first = torch.sum(torch.mul(probs, 1 - probs), dim=1)
                if args.phi_norm:
                    second_last_outputs = second_last_outputs / (second_last_outputs.norm(dim=1, keepdim=True) + 1e-8)
                hessian_second = torch.sum(torch.mul(second_last_outputs, second_last_outputs), dim=1)
                hessian = torch.mul(hessian_first, hessian_second)
                weights_norm = torch.linalg.norm(model.score.weight)
                sharpness = weights_norm * hessian
                sharpness2 = weights_norm * weights_norm * hessian

        loss.backward()
        optimizer.step()

        if args.weight_cap:
            with torch.no_grad():
                if args.cap_type == "spectral":
                    _ = spectral_cap(model.score.weight, s_max=5.0, n_iter=1)
                elif args.cap_type == "frob":
                    _ = frobenius_cap(model.score.weight, args.f_max_norm)

        run_loss += loss.item() * batch["input_ids"].size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

        # collect results here
        with torch.no_grad():
            weights_norm_list.append(weights_norm)
            hessian_list.append(hessian)
            hessian_first_list.append(hessian_first)
            hessian_second_list.append(hessian_second)
            sharpness_list.append(sharpness)
            sharpness2_list.append(sharpness2)

    return (
        run_loss / total,
        correct / total,
        torch.mean(torch.cat(sharpness_list)),
        torch.mean(torch.cat(sharpness2_list)),
        torch.mean(torch.cat(hessian_list)),
        torch.mean(torch.cat(hessian_first_list)),
        torch.mean(torch.cat(hessian_second_list)),
        torch.mean(torch.tensor(weights_norm_list))
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="sst5", choices=["sst2", "sst5"])
    ap.add_argument("--model_name", type=str, default="distilgpt2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--dropout", type=float, default=0.0, help="hidden/attn dropout")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--output_dir", type=str, default=".")
    # ---- regularizer / extras ----
    ap.add_argument("--use_regulation", type=int, default=1)
    ap.add_argument("--lmbd", type=float, default=1e-4)
    ap.add_argument("--training_date", type=str, default="0211")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--phi_norm", type=int, default=1)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--weight_cap", type=int, default=1)
    ap.add_argument("--cap_type", type=str, default="frob")
    ap.add_argument("--f_max_norm", type=float, default=80.0)
    ap.add_argument("--head_only", type=int, default=0)
    ap.add_argument("--train_subset", type=int, default=0, help="limit training set size (0=use full)")
    ap.add_argument("--bump_x", type=float, default=5.0)
    ap.add_argument("--cancel_epoch", default=10000, type=int, help="which epoch to cancel the regularization")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # create the output dir
    args.output_dir = "GPT_{}_Seed{}_Date{}_EPOCH{}_CANCEL{}_BUMP{}_BSZ{}_TSZ{}_LR{}_HD{}_DP{}_WD{}_REGR{}_PHIN{}_LAMBDA{}_TAU{}_WCAP{}_CAPT{}_FMAX{}".format(
        args.task.upper(),
        args.seed,
        args.training_date,
        args.epochs,
        args.cancel_epoch,
        args.bump_x,
        args.batch_size,
        args.train_subset,
        args.lr,
        args.head_only,
        args.dropout,
        args.weight_decay,
        args.use_regulation,
        args.phi_norm,
        args.lmbd,
        args.tau,
        args.weight_cap,
        args.cap_type,
        args.f_max_norm
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print("----- Arguments -----")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("---------------------")

    # ----- Dataset: SST-2 (GLUE) or SST-5 (SetFit/sst5) -----
    if args.task == "sst2":
        ds = load_dataset("glue", "sst2")
        TEXT_COL = "sentence"
        LABEL_COL = "label"
        num_labels = 2
    else:
        ds = load_dataset("SetFit/sst5")  # confirmed correct dataset
        TEXT_COL = "text"
        LABEL_COL = "label"
        num_labels = 5
        # make a validation split if missing
        if "validation" not in ds:
            split = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
            ds["train"] = split["train"]
            ds["validation"] = split["test"]
    
    if args.train_subset > 0:
            ds["train"] = ds["train"].select(range(args.train_subset))
    
    # standardize label column name to 'labels'
    if "labels" not in ds["train"].column_names:
        ds["train"] = ds["train"].rename_column(LABEL_COL, "labels")
    if "labels" not in ds["validation"].column_names:
        ds["validation"] = ds["validation"].rename_column(LABEL_COL, "labels")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def preprocess(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=args.max_length, padding=False)

    # remove all columns except 'labels' after tokenization
    cols_to_remove_train = [c for c in ds["train"].column_names if c != "labels"]
    cols_to_remove_val   = [c for c in ds["validation"].column_names if c != "labels"]

    train_ds = ds["train"].map(preprocess, batched=True, remove_columns=cols_to_remove_train)
    val_ds   = ds["validation"].map(preprocess, batched=True, remove_columns=cols_to_remove_val)


    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=args.num_workers, pin_memory=True)

    # Model (num_labels via config; dropout configurable)

    
    config = GPT2Config.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        output_hidden_states=True,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        resid_pdrop=args.dropout
    )
    model = GPT2ForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Optimizer (head-only or full FT)
    if args.head_only:
        print("tuning head")
        for p in model.transformer.parameters():
            p.requires_grad = False
        optimizer = torch.optim.SGD(
            model.score.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

    # some preparation for regularizer cancellation.
    BASE_LR       = args.lr
    CANCEL_EPOCH  = args.cancel_epoch     # your request
    WD_AFTER      = 1e-4
    BUMP_X        = args.bump_x           # 0.01 -> 0.02
    ETA_MIN       = 5e-5
    
    scheduler = None
    has_cancelled = False

    # Train + eval each epoch
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
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

        (tr_loss, tr_acc, sharpness, sharpness2, hessian,
         hessian_first, hessian_second, weights_norm) = train_one_epoch(
            model, train_loader, optimizer, device, args
        )
        va = eval_loop(model, val_loader, device, args, desc="Val")
        print(f" Train | loss: {tr_loss:.4f} acc: {tr_acc:.4f}")
        print(f" Val   | loss: {va['loss']:.4f} acc: {va['acc']:.4f}")

        if scheduler is not None:
            scheduler.step()
        # save results:
        np.save(args.output_dir + "/sharpness_" + str(epoch), sharpness.detach().cpu().numpy())
        np.save(args.output_dir + "/sharpness2_" + str(epoch), sharpness2.detach().cpu().numpy())
        np.save(args.output_dir + "/hessian_" + str(epoch), hessian.detach().cpu().numpy())
        np.save(args.output_dir + "/hessian_first_" + str(epoch), hessian_first.detach().cpu().numpy())
        np.save(args.output_dir + "/hessian_second_" + str(epoch), hessian_second.detach().cpu().numpy())
        np.save(args.output_dir + "/weights_norm_" + str(epoch), weights_norm.numpy())
        np.save(args.output_dir + "/train_acc_" + str(epoch), tr_acc)
        np.save(args.output_dir + "/train_loss_" + str(epoch), tr_loss)
        np.save(args.output_dir + "/val_loss_" + str(epoch), va['loss'])
        np.save(args.output_dir + "/val_acc_" + str(epoch), va['acc'])
        print("-" * 50)
        print(datetime.now())

if __name__ == "__main__":
    main()
    print("Training done")
