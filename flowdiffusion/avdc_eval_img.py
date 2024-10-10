from img_goal_diffusion import GoalGaussianDiffusion, Trainer
from img_unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from datasets_avdc import CustomSequentialDataset
from torch.utils.data import Subset
import argparse
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import einops
import torch

import sys
sys.path.insert(0,'/home/gary/wm_robot')

from datasets.pusht_dset import PushTDataset

from env.pusht.pusht_wrapper import PushTWrapper

import gym

env = gym.make('pusht', with_velocity=False, with_target=True) # here also make sure that the kwargs are the same!!!
# make sure that the random seed is 99!!!
rand_init_state = env.sample_random_state((99 + 999) * 10)
rand_goal_state = env.sample_random_state(99)
obs_0, state_dct_0 = env.prepare(99, rand_init_state) # obs_0: [224,224,3]
obs_g, state_dct_g = env.prepare(99, rand_goal_state)

epoch_num_to_load = 1 # model-{epoch_num_to_load}.pt file
results_folder = "../results/pusht" # path to save the results and should be like this ""../results/ours""
num_saved_images = 10

valid_n = 1
sample_per_seq = 8
target_size = (128, 128)

base_dataset = PushTDataset(data_path = '/data/jeff/workspace/pusht_dataset', with_velocity=False,n_rollout=5)

train_set = valid_set = [None]

unet = Unet()
pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
text_encoder.requires_grad_(False)
text_encoder.eval()
vision_encoder = CLIPVisionModel.from_pretrained(pretrained_model)
vision_encoder.requires_grad_(False)
vision_encoder.eval()

diffusion = GoalGaussianDiffusion(
    channels=3 * (sample_per_seq - 1),
    model=unet,
    image_size=target_size,
    timesteps=100,
    sampling_timesteps=100,
    loss_type="l2",
    objective="pred_v",
    beta_schedule="cosine",
    min_snr_loss_weight=True,
)

trainer = Trainer(
    diffusion_model=diffusion,
    tokenizer=tokenizer,
    text_encoder=vision_encoder,
    train_set=train_set,
    valid_set=valid_set,
    train_lr=1e-4,
    train_num_steps=60000,
    save_and_sample_every=2500,
    ema_update_every=10,
    ema_decay=0.999,
    train_batch_size=16,
    valid_batch_size=32,
    gradient_accumulate_every=1,
    num_samples=valid_n,
    results_folder=results_folder,
    fp16=True,
    amp=True,
)

trainer.load(epoch_num_to_load)


from PIL import Image
from torchvision import transforms
import torch


transform = transforms.Compose(
    [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
)

obs = obs_0['visual']
obs = [obs]

goal_img = obs_g['visual']
goal_img = einops.rearrange(goal_img, 'h w c -> c h w') / 255.0
goal_img = torch.from_numpy(goal_img)
goal_img = goal_img.unsqueeze(0)

guidance_weight = 0
batch_size = 1

image = transform(to_pil_image(obs[0])) # obs[0] is the first frame of the sequence

output = trainer.sample(
    image.unsqueeze(0), [goal_img], batch_size, guidance_weight
).cpu()

output = output[0].reshape(-1, 3, *target_size)

output = torch.cat([image.unsqueeze(0), output], dim=0)

output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype(
    "uint8"
)

total_width = output.shape[0] * target_size[0]
height = target_size[0]

combined_image = Image.new('RGB', (total_width, height))

for index, img in enumerate(output):
    img_pil = Image.fromarray(img)
    x_offset = index * 128
    combined_image.paste(img_pil, (x_offset, 0))
combined_image.save(f'{results_folder}/inference_result_final.png', format='PNG')
    
# for idx in tqdm(range(num_saved_images)):
#     random_idx = random.randint(0, len(base_dataset) - 1)
#     obs, act, state, mask = base_dataset[random_idx]
#     obs = obs['visual']
    
#     goal_img = obs[10].unsqueeze(0)

#     guidance_weight = 0
#     batch_size = 1

#     image = transform(to_pil_image(obs[0])) # obs[0] is the first frame of the sequence

#     output = trainer.sample(
#         image.unsqueeze(0), [goal_img], batch_size, guidance_weight
#     ).cpu()

#     output = output[0].reshape(-1, 3, *target_size)

#     output = torch.cat([image.unsqueeze(0), output], dim=0)

#     output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype(
#         "uint8"
#     )

#     total_width = output.shape[0] * target_size[0]
#     height = target_size[0]

#     combined_image = Image.new('RGB', (total_width, height))

#     for index, img in enumerate(output):
#         img_pil = Image.fromarray(img)
#         x_offset = index * 128
#         combined_image.paste(img_pil, (x_offset, 0))
#     combined_image.save(f'{results_folder}/inference_result_{idx}.png', format='PNG')