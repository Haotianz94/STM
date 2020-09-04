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
import json

from dataset_tennis import TennisDataset
from model import STM

torch.set_grad_enabled(False) # Volatile

##############################################################################
# Main function
##############################################################################

def run_video(dataset, MFs, Ms, num_objects, num_frames, output_dir):

    window = 5

    num_masks = Ms.shape[1]

    # Save first mask frame
    # result_shape = [1] + list(Fs.shape)
    # result_shape[1] = Ms.shape[0]
    # result_shape = torch.Size(result_shape)
    # Es = torch.zeros(result_shape) # 1 x K x N X H X W 
    # print(result_shape)

    # Fs = torch.from_numpy(Fs).unsqueeze(0).cuda()
    MFs = torch.from_numpy(MFs).unsqueeze(0).cuda()
    Ms = torch.from_numpy(Ms).unsqueeze(0).cuda()

    mem_key = None
    mem_val = None
    for i in range(num_masks):
        with torch.no_grad():
            curr_key, curr_val = model(MFs[:,:,i], Ms[:,:,i], torch.tensor([1]))
        if mem_key is None:
            mem_key = curr_key
            mem_val = curr_val
        else:
            mem_key = torch.cat([mem_key, curr_key], dim=3) # 1 x K x C x N x H x W 
            mem_val = torch.cat([mem_val, curr_val], dim=3)

    for t in tqdm.tqdm(range(num_frames)):

        frame_input, frame_output_id = dataset.get_next_frame()

        # segment from memory
        with torch.no_grad():
            frame_input = torch.from_numpy(frame_input).unsqueeze(0).cuda() # 1 x 3 x H x W
            logit = model(frame_input, mem_key, mem_val, torch.tensor([num_objects]))
            output = F.softmax(logit, dim=1)

        with torch.no_grad():
            curr_key, curr_val = model(frame_input, output, torch.tensor([1]))

        mem_key = torch.cat([mem_key, curr_key], dim=3)
        mem_val = torch.cat([mem_val, curr_val], dim=3)

        # Keep a rolling window of size 4 + first frame in memory
        # Remove earliest frame (except first) from memory
        if mem_key.shape[3] > window:
            first_key, head_key, tail_keys = torch.split(mem_key, [num_masks, 1, window - num_masks], dim=3)
            first_val, head_val, tail_vals = torch.split(mem_val, [num_masks, 1, window - num_masks], dim=3)
            del head_key, head_val
            mem_key = torch.cat([first_key, tail_keys], dim=3)
            mem_val = torch.cat([first_val, tail_vals], dim=3)

        mask = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)
        # Fill back cropped region
        if dataset.infer_mask_fg:
            mask = np.concatenate((dataset.mask_filling_fg, mask), axis=0)
        else:
            mask = np.concatenate((mask, dataset.mask_filling_bg), axis=0)

        mask_pil = Image.fromarray(255 * mask)
        mask_pil.save(os.path.join(output_dir, '{:08d}.png'.format(frame_output_id)))

        del frame_input
        del output
        torch.cuda.empty_cache()

##############################################################################
# Setup model
##############################################################################

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN
cp = torch.load("STM_weights.pth")
model.load_state_dict(cp)


video_id = 16
video_table = json.load(open('/home/haotian/Projects/racket2game/db/video_table.json', 'r'))
video = video_table[video_id]

##############################################################################
# Run model on video
##############################################################################

for side in ['B']:
    video_path = video['path']
    point_bounds = [b for pid, b in enumerate(video['point_bounds']) if (side == 'A') ^ (pid not in video['points_foreground'])]

    for infer_mask_fg in [False, True]:
        dataset = TennisDataset(os.path.join("datasets", '{:03}_{}'.format(video_id, side)), video_path, point_bounds, infer_mask_fg)
        while True:
            next_point = dataset.get_next_point()
            if next_point == -1:
                break
            if next_point == 0:
                continue
            MFs, Ms, num_objects, num_frames = next_point

            output_dir = os.path.join('./results',  '{:03}_{}'.format(video_id, 'fg' if infer_mask_fg else 'bg'))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Run model
            run_video(dataset, MFs, Ms, num_objects, num_frames, output_dir)

            del MFs, Ms