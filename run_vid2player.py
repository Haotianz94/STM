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

from dataset_tennis import TennisDataset
from model import STM

torch.set_grad_enabled(False) # Volatile

video_id = 15

##############################################################################
# Main function
##############################################################################

def run_video(Fs, MFs, Ms, num_objects):

    window = 5

    num_frames = Fs.shape[1]
    num_masks = Ms.shape[1]

    # Save first mask frame
    result_shape = [1] + list(Fs.shape)
    result_shape[1] = Ms.shape[0]
    result_shape = torch.Size(result_shape)
    Es = torch.zeros(result_shape) # 1 x K x N X H X W 
    print(result_shape)

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

    pred = []
    for t in tqdm.tqdm(range(num_frames)):

        # segment from memory
        with torch.no_grad():
            frame = torch.from_numpy(Fs[:,t]).unsqueeze(0).cuda()
            logit = model(frame, mem_key, mem_val, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)

        with torch.no_grad():
            curr_key, curr_val = model(frame, Es[:,:,t], torch.tensor([1]))

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

        del frame
        torch.cuda.empty_cache()
        # pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
        # pred.append(torch.argmax(Es[0, :, t], dim=0).cpu().numpy().astype(np.uint8))

    print("Running argmax over outputs ...")
    # # pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    pred = torch.argmax(Es[0], dim=0).cpu().numpy().astype(np.uint8)
    return pred


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
for side in ['A', 'B']:
    dataset = TennisDataset(os.path.join("datasets", '{:03}_{}'.format(video_id, side)))
    for fg in [True, False]:
        dataset.infer_mask_fg = fg
        while True:
            next_point = dataset.get_next_point()
            if next_point == -1:
                break
            if next_point == 0:
                continue
            Fs, MFs, Ms, num_objects, info = next_point

            num_frames = info['num_frames']
            point_id = info['point_id']

            print("Running {}th point".format(point_id))

            # Run model
            pred = run_video(Fs, MFs, Ms, num_objects)
                
            # Save results
            test_path = os.path.join('./results',  '{:03}_{}'.format(video_id, 'fg' if fg else 'bg'))
            print("Dumping results for {}th point".format(point_id))
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            for i in range(num_frames):
                img_E = Image.fromarray(255 * pred[i])
                img_E.save(os.path.join(test_path, '{:08d}.png'.format(info['frame_ids'][i])))

            del Fs, MFs, Ms, pred