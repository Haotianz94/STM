import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tqdm
import os
import sys
import time

from dataset_custom import CustomDataset, CustomDatasetMultiMask
from model import STM

torch.set_grad_enabled(False) # Volatile

DATASET = "test2"
MULTIMASK = False

##############################################################################
# Main function
##############################################################################

def run_video(Fs, Ms, num_frames, num_objects):

    window = 5

    # Save first mask frame
    result_shape = list(Fs.shape)
    result_shape[1] = Ms.shape[1]
    result_shape = torch.Size(result_shape)
    Es = torch.zeros(result_shape) # 1 x K x N X H X W 
    # Es = torch.zeros_like(Ms)

    Es[:,:,0] = Ms[:,:,0]

    with torch.no_grad():
        mem_key, mem_val = model(Fs[:,:,0], Es[:,:,0], torch.tensor([1]))

    for t in tqdm.tqdm(range(1, num_frames)):

        # segment from memory
        with torch.no_grad():
            print(Fs[:,:,t].shape)
            logit = model(Fs[:,:,t], mem_key, mem_val, torch.tensor([num_objects]))
            print(logit.shape)
        Es[:,:,t] = F.softmax(logit, dim=1)

        with torch.no_grad():
            curr_key, curr_val = model(Fs[:,:,t], Es[:,:,t], torch.tensor([1]))

        mem_key = torch.cat([mem_key, curr_key], dim=3)
        mem_val = torch.cat([mem_val, curr_val], dim=3)

        # Keep a rolling window of size 4 + first frame in memory
        # Remove earliest frame (except first) from memory
        if mem_key.shape[3] > window:
            first_key, head_key, tail_keys = torch.split(mem_key, [1, 1, window - 1], dim=3)
            first_val, head_val, tail_vals = torch.split(mem_val, [1, 1, window - 1], dim=3)
            del head_key, head_val
            mem_key = torch.cat([first_key, tail_keys], dim=3)
            mem_val = torch.cat([first_val, tail_vals], dim=3)
    
    print("Running argmax over outputs ...")
    # pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    pred = torch.argmax(Es[0], dim=0).cpu().numpy().astype(np.uint8)
    return pred, Es


def run_video_multimask(Fs, Ms, num_frames, num_objects, mask_ids):

    # Save first mask frame
    result_shape = list(Fs.shape)
    result_shape[1] = Ms.shape[1]
    result_shape = torch.Size(result_shape)
    Es = torch.zeros(result_shape) # 1 x K x N X H X W 

    mem_key = None
    mem_val = None
    for i in range(len(mask_ids)):
        with torch.no_grad():
            curr_key, curr_val = model(Fs[:,:,mask_ids[i][0]], Ms[:,:,i], torch.tensor([1]))
        if mem_key is None:
            mem_key = curr_key
            mem_val = curr_val
        else:
            mem_key = torch.cat([mem_key, curr_key], dim=3)
            mem_val = torch.cat([mem_val, curr_val], dim=3)

    mem_key_update = mem_key
    mem_val_update = mem_val
    for i in tqdm.tqdm(range(num_frames)):
        # Rerun segmentation for input masks
        # if i in mask_ids:
        #     Es[:,:,i] = Ms[:,:,i]
        #     continue

        # segment from memory
        with torch.no_grad():
            logit = model(Fs[:,:,i], mem_key_update, mem_val_update, torch.tensor([num_objects]))
        Es[:,:,i] = F.softmax(logit, dim=1)

        with torch.no_grad():
            curr_key, curr_val = model(Fs[:,:,i], Es[:,:,i], torch.tensor([1]))
        mem_key_update = torch.cat([mem_key, curr_key], dim=3)
        mem_val_update = torch.cat([mem_val, curr_val], dim=3)

        # Keep a rolling window of size 4 + first frame in memory
        # Remove earliest frame (except first) from memory
        # if mem_key.shape[3] > window:
        #     first_key, head_key, tail_keys = torch.split(mem_key, [1, 1, window - 1], dim=3)
        #     first_val, head_val, tail_vals = torch.split(mem_val, [1, 1, window - 1], dim=3)
        #     del head_key, head_val
        #     mem_key = torch.cat([first_key, tail_keys], dim=3)
        #     mem_val = torch.cat([first_val, tail_vals], dim=3)
    
    print("Running argmax over outputs ...")
    # pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    pred = torch.argmax(Es[0], dim=0).cpu().numpy().astype(np.uint8)
    return pred, Es

##############################################################################
# Setup model
##############################################################################

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN
cp = torch.load("STM_weights.pth")
model.load_state_dict(cp)

##############################################################################
# Run model on each video
##############################################################################
if MULTIMASK:
    dataset = CustomDatasetMultiMask(os.path.join("datasets", DATASET))
else:
    dataset = CustomDataset(os.path.join("datasets", DATASET))
dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

for seq, V in enumerate(dataloader):
    Fs, Ms, num_objects, info = V
    num_frames = info['num_frames'][0].item()

    # Run model
    if MULTIMASK:
        pred, Es = run_video_multimask(Fs, Ms, num_frames, num_objects, info['mask_ids'])
    else:
        pred, Es = run_video(Fs, Ms, num_frames, num_objects)
        
    # Save results
    test_path = os.path.join('./results', DATASET)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(255 * pred[f])
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))
