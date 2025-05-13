# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import deepspeed
from loguru import logger

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='target_model_path ')
parser.add_argument('--tmpdir', type=str,
                    default='pre_generated_data ')
parser.add_argument('--cpdir', type=str, default='path_to_save_the_ckpt')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

parser.add_argument('--run_mode', type=str, default='train')  # "train" "debug"
parser.add_argument('--logger_file', type=str, default='default')
parser.add_argument('--resume_from', type=int, default=0)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--mlp_num', type=int, default=5)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--train_mlp_input', type=str, default='decoder_output')  # "ground_truth" "decoder_output"
parser.add_argument('--mlp_loss_weight', type=float, default=9)
parser.add_argument('--only_accept_max_each_epoch', type=int, default=0)
parser.add_argument('--configpath', type=str, default="0")
parser.add_argument('--model_name', type=str, default="l3_8b")
parser.add_argument('--data_noise', type=int, default=1)
parser.add_argument('--p_w', type=float, default=0.1)
parser.add_argument('--v_w', type=float, default=1.0)
parser.add_argument('--mlp_v_w', type=float, default=1.0)
parser.add_argument('--mlp_p_w', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--existing_model_path', type=str)
parser.add_argument('--topk_loss_num', type=int, default=0)
parser.add_argument('--mlp_loss_decay_coefficient', type=float, default=0.8)




parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

logger.info(f"{args=}")
logger.remove()

import json

train_config = {
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 200,
    "p_w": args.p_w,
    "v_w": args.v_w,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": args.max_len,
}

from safetensors import safe_open
import os
import torch
import torch.distributed as dist


torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)

import sys
sys.path.append('./gumiho')

from model.cnets import Model
from model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


deepspeed.init_distributed()
rank = torch.distributed.get_rank()

if rank == 0:
    import wandb
    wandb.init(project="wandb_proj", name=args.logger_file, config=train_config)
    logger.add(f"./{args.model_name}/{args.logger_file}.log", level="DEBUG", mode="w")
    logger.info(f"{train_config = }")
    logger.info(f"{args = }")


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

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor

for param in head.parameters():
    param.requires_grad = False


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path, followlinks=True):
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
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
  
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




def compute_loss(target, target_p, predict, loss_mask, topk_loss_num=0, args=None):

    ploss_mlp, vloss_mlp = 0, 0
    mlp_metric = {}
    predict_mlp = predict[:-1]
    target, target_p = target.to(rank), target_p.to(rank)
    mlp_correct_total, mlp_total = 0, 0

    for idx, predict_i in enumerate(predict_mlp):
        hidden_state_shifted = predict_i[:,:-(2+idx)].contiguous()
        target_shifted = target[:, (2+idx):].contiguous()
        target_p_shifted = target_p[:, (2+idx):].contiguous()
        _loss_mask = loss_mask[:, (2+idx):].contiguous()

        _vloss = criterion(hidden_state_shifted, target_shifted)
        _vloss = torch.sum(torch.mean(_loss_mask * _vloss, 2)) / (_loss_mask.sum() + 1e-5) # 1227 mean->sum

        _out_head = head_engine(hidden_state_shifted)
        _out_logp = nn.LogSoftmax(dim=2)(_out_head)
        _plogp = target_p_shifted * _out_logp
        _ploss = -torch.sum(torch.sum(_loss_mask * _plogp, 2)) / (_loss_mask.sum() + 1e-5)

       
        decay_coefficient = float(args.mlp_loss_decay_coefficient)
        vloss_mlp += (_vloss * (decay_coefficient ** (idx+1)))
        ploss_mlp += (_ploss * (decay_coefficient ** (idx+1)))

        with torch.no_grad():
            _, _predicted = torch.max(_out_head, 2)
            _, _target = torch.max(target_p_shifted, 2)
            _ct = _loss_mask.sum().item()
            _cc = ((_predicted == _target) * _loss_mask.squeeze()).sum().item()
            mlp_correct_total += _cc
            mlp_total += _ct
        
        if _ct > 0:
            mlp_metric[f"mlp{idx}_accu"] = _cc / (_ct + 1e-5)
            logger.info(f"For mlp{idx}: {_cc=}, {_ct=}")
        else:
            logger.info(f"_ct is 0, {_cc=}, {_ct=}")
    
    mlp_metric[f"mlp_vloss"] = vloss_mlp.detach().cpu().item()
    mlp_metric[f"mlp_ploss"] = ploss_mlp.detach().cpu().item()
    mlp_metric[f"mlp_total_accu"] = mlp_correct_total / (mlp_total + 1e-5)


    predict = predict[-1]
    out_head = head_engine(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target.to(rank))

   
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5) # 1227 mean->sum

    if topk_loss_num > 1e-5:
        topk_mask = torch.topk(target_p, k=topk_loss_num, dim=2).indices
        topk_loss = -torch.sum(torch.sum(loss_mask * plogp.gather(dim=2, index=topk_mask), 2)) / (loss_mask.sum() + 1e-5)
        vloss = vloss + topk_loss

    return vloss, ploss, vloss_mlp, ploss_mlp, out_head, mlp_metric



def gather_totals(total):

    if not isinstance(total, torch.Tensor):
        total = torch.tensor(total, dtype=torch.float32)
    total = total.cuda()

    total_list = [torch.zeros_like(total) for _ in range(dist.get_world_size())]

    dist.all_gather(total_list, total)

    return total_list


if args.data_noise == 1:
    logger.info(f"Add data_noise")
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    logger.info(f"No data_noise")
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath))]
testdatapath = datapath[int(len(datapath) * 0.95):]
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
test_loader = DataLoader(testdataset, batch_size=4, shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)


if rank == 0:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(args.configpath)
model = Model(config, path=args.basepath, load_emb=True, args=args)


if args.start_epoch > 0 or args.existing_model_path is not None:
    model_state_dict = torch.load(args.existing_model_path, map_location=f"cuda:{rank}")
    model.load_state_dict(model_state_dict, strict=True)
    logger.info(f"Loading ckpt at {args.existing_model_path}")


criterion = nn.SmoothL1Loss(reduction="none")



model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=args,
                                                                model=model,
                                                                model_parameters=model.parameters(),
                                                                training_data=traindataset,
                                                                collate_fn=DataCollatorWithPadding()
                                                                )


head_engine = head.half().to(rank)
head_engine.eval()

            


num_epochs = train_config["num_epochs"]

mlp_p_w_flag = args.mlp_p_w

for epoch in range(args.start_epoch, num_epochs):
    if mlp_p_w_flag == 100:
        if epoch < 2:
            args.mlp_p_w = 0
        else:
            args.mlp_p_w = 0.1
    elif mlp_p_w_flag == 400:
        if epoch < 2:
            args.mlp_p_w = 0
        else:
            args.mlp_p_w = 0.4

    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    original_loss = 0

    if rank == 0:
        tqdm_desc = f"Epoch {epoch}"
        epoch_iterator = tqdm(train_loader, desc=tqdm_desc)
    else:
        epoch_iterator = train_loader 
    
    if args.run_mode == "debug" and epoch == 1:
        raise ValueError(f"Successful at epoch == 1")

    for batch_idx, data in enumerate(epoch_iterator):
        if args.run_mode == "debug" and batch_idx == 2:
            logger.info(f"Break at batch_idx == 2")
            break

        model_engine.zero_grad()

        predict = model_engine(data["hidden_states"].to(rank).half(), input_ids=data["input_ids"].to(rank),
                               attention_mask=data["attention_mask"].to(rank).half())
        with torch.no_grad():
            target_head = head_engine(data["target"].to(rank).half())
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        loss_mask = data["loss_mask"][:, :, None].to(rank)
        vloss, ploss, vloss_mlp, ploss_mlp, out_head, mlp_metric = compute_loss(data["target"], target_p, predict, loss_mask, args.topk_loss_num, args)
        if args.mlp_p_w == 0:
            loss = args.v_w * vloss + args.p_w * ploss + args.mlp_v_w * vloss_mlp/args.mlp_loss_weight
        else:
            loss = args.v_w * vloss + args.p_w * ploss + args.mlp_v_w * vloss_mlp/args.mlp_loss_weight + args.mlp_p_w * ploss_mlp/args.mlp_loss_weight

        model_engine.backward(loss)
        model_engine.step()


        with torch.no_grad():
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
        if rank == 0 and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/mlp_ploss": ploss_mlp.item(), "train/mlp_vloss": vloss_mlp.item(), "train/loss": loss.item(), "train/acc": cc / ct, 
                       "avg_acc":  correct / (total + 1e-5)}
            
            wandb.log(logdict)
            wandb.log(mlp_metric)
            logger.info(f"Epoch{epoch} Batch{batch_idx}: {logdict}")
            logger.info(f"Epoch{epoch} Batch{batch_idx}: {mlp_metric}")
            

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    
    epoch_loss /= num_batches
    
    if rank == 0:
         
        logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        logger.info(f"{epoch=}, {correct=}, {total=}")
        logger.info('Train Accuracy: {:.2f}%'.format(100 * correct / (total + 1e-5)))
        wandb.log({"train/epochacc": correct / (total + 1e-5), "train/epochloss": epoch_loss, "epoch": epoch})
        
    if args.run_mode == "train":
        model_engine.save_16bit_model(f"{args.cpdir}/state_{epoch}")
    
