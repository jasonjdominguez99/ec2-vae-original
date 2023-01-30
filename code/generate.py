# generate.py
#
# source code for generating music, using a trained
# EC2-VAE model

# imports
import json
import torch
import numpy as np
from torch.distributions import Normal
from ec_squared_vae import ECSquaredVAE
from data_loader import MusicArrayLoader

# function definitions
def load_ec_squared_vae(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)
    
    load_path = "ec_squared_vae/params/{}.pt".format(args["name"])

    model = ECSquaredVAE(
        args["roll_dim"], args["hidden_dim"], args["rhythm_dim"], 
        args["condition_dims"], args["z1_dim"],
        args["z2_dim"], args["time_step"]
    )
    # remove module. from start of state dict keys
    from collections import OrderedDict
    state_dict = torch.load(load_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)

    return model

def get_latent_dist(model, data_path):
    # Code from main
    dl = MusicArrayLoader(data_path, 32, 16)
    dl.chunking()
    
    batch, c = dl.get_batch(1)
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

    dis1, dis2 = model.encoder(encode_tensor, c)
    z1 = dis1.rsample()
    z2 = dis2.rsample()
    
    return z1, z2, c

def main():
    # Load trained model
    config_file_path = "ec_squared_vae/code/ec_squared_vae_model_config.json"
    model = load_ec_squared_vae(config_file_path)
    model.training = False
    print("Loaded!")
    
    # Encode the source and target melodies to get z_p of the source and
    # z_r of the target 
    generation_config_file_path = "ec_squared_vae/code/generation_config.json"
    with open(generation_config_file_path) as f:
        args = json.load(f)
        
    z_p, _, chords = get_latent_dist(model, args["source"])
    _, z_r, _ = get_latent_dist(model, args["target"])
    print("Encoded!")
    
    # Decode the obtained z_p and z_r to show control, generating
    # a melody with pitches of source and rhythm of target
    result = model.decoder(z_p, z_r, chords).detach().numpy()[0]
    print("Decoded!")
    
    print(result[0])
    print(len(result[0]))
    
    print(result.shape)
    print(result)
    
if __name__ == "__main__":
    main()
