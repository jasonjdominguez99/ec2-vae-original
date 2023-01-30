# main.py
#
# main source code for training and saving the
# EC^2 VAE model


# imports
import json
import os

from ec_squared_vae import ECSquaredVAE
from utils import (
    MinExponentialLR, loss_function
)
from data_loader import MusicArrayLoader

import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter


# function definitions and implementations
def configure_model(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)

    if not os.path.isdir("ec_squared_vae/log"):
        os.mkdir("ec_squared_vae/log")

    if not os.path.isdir("ec_squared_vae/params"):
        os.mkdir("ec_squared_vae/params")

    save_path = "ec_squared_vae/params/{}.pt".format(args["name"])
    writer = SummaryWriter("ec_squared_vae/log/{}".format(args["name"]))

    model = ECSquaredVAE(
        args["roll_dim"], args["hidden_dim"], args["rhythm_dim"], 
        args["condition_dims"], args["z1_dim"],
        args["z2_dim"], args["time_step"]
    )

    if args["if_parallel"]:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    if args["decay"] > 0:
        scheduler = MinExponentialLR(
            optimizer, gamma=args["decay"], minimum=1e-5
        )

    if torch.cuda.is_available():
        print(
            "Using: ",
            torch.cuda.get_device_name(torch.cuda.current_device())
        )
        model.cuda()
    else:
        print("CPU mode")

    step, pre_epoch = 0, 0
    model.train()

    dl = MusicArrayLoader(args["data_path"], args["time_step"], 16)
    dl.chunking()
    
    dl_val = MusicArrayLoader(args["val_data_path"], args["time_step"], 16)
    dl_val.chunking()

    return (model, args, save_path, writer, 
            scheduler, step, pre_epoch, dl, 
            dl_val, optimizer)
    

def prepare_batch(batch, c):
    encode_tensor = torch.from_numpy(batch).float()
    c = torch.from_numpy(c).float()

    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(
        -1, rhythm_target.size(-1)
    ).max(-1)[1]
    target_tensor = encode_tensor.view(
        -1, encode_tensor.size(-1)
    ).max(-1)[1]

    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
        rhythm_target = rhythm_target.cuda()
        c = c.cuda()
        
    return encode_tensor, target_tensor, rhythm_target, c


def train(model, args, writer, scheduler, step, dl, optimizer):
    batch, c = dl.get_batch(args["batch_size"])
    encode_tensor, target_tensor, rhythm_target, c = prepare_batch(batch, c)

    optimizer.zero_grad()
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
    distribution_1 = Normal(dis1m, dis1s)
    distribution_2 = Normal(dis2m, dis2s)

    loss = loss_function(
        recon,
        recon_rhythm,
        target_tensor,
        rhythm_target,
        distribution_1,
        distribution_2,
        step,
        beta=args["beta"]
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1

    print("batch loss: {:.5f}".format(loss.item()))
    writer.add_scalar("batch_loss", loss.item(), step)
    if args["decay"] > 0:
        scheduler.step()
    dl.shuffle_samples()

    return step


def validate(model, args, writer, step, dl_val):
    batch, c = dl_val.get_batch(dl_val.get_n_sample())
    encode_tensor, target_tensor, rhythm_target, c = prepare_batch(batch, c)
    
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
    distribution_1 = Normal(dis1m, dis1s)
    distribution_2 = Normal(dis2m, dis2s)
    
    loss = loss_function(
        recon,
        recon_rhythm,
        target_tensor,
        rhythm_target,
        distribution_1,
        distribution_2,
        step,
        beta=args["beta"]
    )
    
    print("validation loss: {:.5f}".format(loss.item()))
    writer.add_scalar("val_loss", loss.item(), step)


def main():
    config_fname = "ec_squared_vae/code/ec_squared_vae_model_config.json"

    (model, args, save_path, writer, scheduler,
     step, pre_epoch, dl, dl_val, optimizer) = configure_model(config_fname)

    with tqdm(total=args["n_epochs"]) as epochs_pbar:
        while dl.get_n_epoch() < args["n_epochs"]:
            step = train(model, args, writer, scheduler, step, dl, optimizer)
            if dl.get_n_epoch() != pre_epoch:
                pre_epoch = dl.get_n_epoch()
                torch.save(model.cpu().state_dict(), save_path)

                if torch.cuda.is_available():
                    model.cuda()
                    
                validate(model, args, writer, step, dl_val)
                print("Model saved!")
                epochs_pbar.update(1)

if __name__ == "__main__":
    main()
