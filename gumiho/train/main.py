# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
os.environ['MIOPEN_DISABLE_CACHE'] = '1'
import argparse
from loguru import logger
import shutil

import sys
import torch.distributed as dist

import gc
from addict import Dict



parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='target_model_path ')
parser.add_argument('--configpath', type=str, default="vicuna_7B_config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='pre_generated_data ')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--run_mode', type=str, default='train')  # "train" "debug"
parser.add_argument('--logger_file', type=str, default='default')
parser.add_argument('--resume_from', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--train_mlp_input', type=str, default='decoder_output')
parser.add_argument('--mlp_loss_weight', type=float, default=1.0)
parser.add_argument('--only_accept_max_each_epoch', type=int, default=0)
parser.add_argument('--total_step', type=float, default=8.0)
parser.add_argument('--warmup_step', type=int, default=6000)
parser.add_argument('--min_lr_rate', type=float, default=0)


args = parser.parse_args()

if args.run_mode == "train":
    logger_level = "INFO"
elif args.run_mode == "debug":
    logger_level = "DEBUG"
    os.environ['WANDB_MODE'] = 'disabled'

logger.remove()


train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": args.warmup_step,
    "total_steps": args.total_step*100000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}



import json
from safetensors import safe_open
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = False
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
from ..model.cnets import Model
from ..model.configs import EConfig
from typing import Any, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig


if accelerator.is_main_process:
    import wandb

    wandb.init(project="wandb_proj", name=args.logger_file, config=train_config)

    logger.add(f"./{args.model_name}/{args.logger_file}.log", level=logger_level, mode="w")
    logger.info(f"{train_config = }")
    logger.info(f"{args = }")

torch.set_printoptions(threshold=float('inf'))

baseconfig = AutoConfig.from_pretrained(args.basepath)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features):
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask, accum_step=1):
    ploss_mlp, vloss_mlp = 0, 0
    mlp_metric = {}
    predict_mlp = predict[:-1]
    for idx, predict_i in enumerate(predict_mlp):
        logger.debug(f"{idx = }")
        hidden_state_shifted = predict_i[:,:-(2+idx)].contiguous()
        target_shifted = target[:, (2+idx):].contiguous()
        target_p_shifted = target_p[:, (2+idx):].contiguous()
        _loss_mask = loss_mask[:, (2+idx):].contiguous()

        _vloss = criterion(hidden_state_shifted, target_shifted)
        _vloss = torch.sum(torch.mean(_loss_mask * _vloss, 2)) / (_loss_mask.sum() + 1e-5)

        _out_head = head(hidden_state_shifted)
        _out_logp = nn.LogSoftmax(dim=2)(_out_head)
        _plogp = target_p_shifted * _out_logp
        _ploss = -torch.sum(torch.sum(_loss_mask * _plogp, 2)) / (_loss_mask.sum() + 1e-5)

        vloss_mlp += _vloss
        ploss_mlp += _ploss

        mlp_metric[f"mlp{idx}_loss"] = _vloss + _ploss * 0.1
        mlp_metric[f"mlp{idx}_outhead"] = _out_head


        logger.debug(f"{vloss_mlp = }, {ploss_mlp = }")


    predict = predict[-1]
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)

    vloss = (vloss + vloss_mlp/args.mlp_loss_weight) * accum_step
    ploss = (ploss + ploss_mlp/args.mlp_loss_weight) * accum_step

    return vloss, ploss, out_head, mlp_metric

@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=max_length, use_cache=True):
        
       
        mlp_input_hidden_states, mlp_input_ids = [], []
    
        if use_cache:
            past_key_values = None
            for i in range(2):  
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values,
                                                        use_cache=True, mode="gumiho_generate")
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True, mode="gumiho_generate")
                last_hidden = out_hidden[:, -1:]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)

                mlp_input_hidden_states.append(last_hidden)
                mlp_input_ids.append(token)


        else:
            raise NotImplementedError

        mlp_input_hidden_states = torch.cat(mlp_input_hidden_states, dim=1)  # torch.Size([4, 3, 4096])
        mlp_input_ids = torch.cat(mlp_input_ids, dim=1)  # torch.Size([4, 3])


        mlp_output_hidden = model(mlp_input_hidden_states, input_ids=mlp_input_ids, mode="mlp_generate")
        for hidden_i in mlp_output_hidden:
            mlp_headout = head(hidden_i)
            mlp_token = torch.argmax(mlp_headout, dim=-1)

            input_ids = torch.cat((input_ids, mlp_token), dim=1)

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


class CustomLRScheduler:
    def __init__(self, optimizer, warmup_steps: int, max_lr: float, min_lr_factor: float, decay_steps: int):
        
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_factor
        self.decay_steps = decay_steps
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        elif self.current_step < self.warmup_steps + self.decay_steps:
            decay_step = self.current_step - self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * max(0, (self.decay_steps - decay_step) / self.decay_steps)
        else:
            lr = self.min_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(config, load_emb=True, path=args.basepath, args=args)


criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    if args.min_lr_rate < 1e-5:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)
    else:
        scheduler = CustomLRScheduler(optimizer, num_warmup_steps, train_config["lr"], args.min_lr_rate, total_steps)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

start_epoch = args.resume_from
if start_epoch > 0:
    logger.info(f"resume from {start_epoch}")
    accelerator.load_state(f"{args.cpdir}/temp")
    


max_saved = 0
max_mean_accept_token = 0
for epoch in range(start_epoch, num_epochs + 1):
    

    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    if accelerator.is_main_process:
        tqdm_desc = f"Epoch {epoch}"
        logger.info(tqdm_desc)
        epoch_iterator = tqdm(train_loader, desc=tqdm_desc)
    else:
        epoch_iterator = train_loader 
        
    for batch_idx, data in enumerate(epoch_iterator):
        
        if args.run_mode == "debug":
            if batch_idx > 3:
                logger.debug("args.run_mode == debug, now break")
                break

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            
            with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            loss_mask = data["loss_mask"][:, :, None]
            vloss, ploss, out_head, mlp_metric = compute_loss(data["target"], target_p, predict, loss_mask, train_config["gradient_accumulation_steps"])
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss

            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            origin_target = target
            logger.debug(f"Origin loss mask: {loss_mask.shape = }")
            ct = loss_mask.sum().item()
            logger.debug(f"Origin loss mask ct: {ct = }")
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc

            if ct != 0:
                mlp_logdict = {}
                for idx in range(args.mlp_num):
                    _mlp_head = mlp_metric[f"mlp{idx}_outhead"]
                    _, _mlp_predicted = torch.max(_mlp_head, 2)
                    _loss_mask_shift = loss_mask[:, (2+idx):]
                    logger.debug(f"Shifted loss mask shape: {_loss_mask_shift.shape = }")
                    logger.debug(f"{origin_target.shape = }")
                    _target_shift = origin_target[:, (2+idx):]
                    _mlp_predicted_shift = _mlp_predicted
                    _ct = _loss_mask_shift.sum().item()
                    logger.debug(f"Shifted loss mask ct: {_ct = }")
                    logger.debug(f"{_mlp_predicted_shift.shape = }")
                    logger.debug(f"{_mlp_head.shape = }")
                    logger.debug(f"{_target_shift.shape = }")
                    _cc = ((_mlp_predicted_shift == _target_shift) * _loss_mask_shift.squeeze()).sum().item()
                    mlp_logdict[f"train/mlp{idx}_acc"] = _cc / _ct
            
            # ===============================




        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            for i in range(args.mlp_num):
                logdict[f'train/mlp{i}_loss'] = mlp_metric[f"mlp{i}_loss"]
            
            wandb.log(logdict)
            wandb.log(mlp_logdict)
            logger.info(f"{logdict = }")
            logger.info(f"{mlp_logdict = }")


        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})
        accelerator.save_state(output_dir=f"{args.cpdir}/epoch{epoch}")


    if (epoch+1) % args.test_freq == 0:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()
        max_length = args.mlp_num + 2

        k_acc = [[] for i in range(max_length)]
        for batch_idx, data in enumerate(tqdm(test_loader)):
            if args.run_mode == "debug":
                if batch_idx > 2:
                    logger.debug("args.run_mode == debug, now break")
                    break

            with torch.no_grad():
                
                if batch_idx < 10:
                    acces = getkacc(model, data, head, max_length=max_length)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head, mlp_metric = compute_loss(data["target"], target_p, predict, loss_mask, train_config["gradient_accumulation_steps"])
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            
