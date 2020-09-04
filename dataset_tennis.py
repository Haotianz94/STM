import os
import cv2
import numpy as np
from PIL import Image
import torch


class TennisDatasetImage(object):
    """
    Generate player masks for Vid2player, load frames from decoded images
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.K = 2
        self.mask_fg = sorted([f for f in os.listdir(os.path.join(root_dir, 'mask_fg'))])
        self.mask_bg = sorted([f for f in os.listdir(os.path.join(root_dir, 'mask_bg'))])
        self.point_dir = sorted([f for f in os.listdir(root_dir) if not f.startswith('m')])
        self.num_points = len(self.point_dir)

        self.infer_mask_fg = True
        self.point_id = 0

        # Crop image to reduce computation
        self.fg_cutline = 300
        self.bg_cutline = 500
        self.mask_filling_fg = np.zeros((self.fg_cutline, 1920), dtype=np.uint8)
        self.mask_filling_bg = np.zeros((1080 - self.bg_cutline, 1920), dtype=np.uint8)


    def get_next_point(self):
        if self.point_id == self.num_points:
            self.point_id = 0
            return -1

        point_output_id = int(self.point_dir[self.point_id])
        self.frame_names = sorted([f for f in os.listdir(os.path.join(self.root_dir, self.point_dir[self.point_id]))])
        self.frame_output_ids = sorted([int(f[:-4]) for f in self.frame_names])
        self.frame_id = 0

        if self.infer_mask_fg:
            Ms = [self.get_image(os.path.join(self.root_dir, 'mask_fg', f), mask=True) for f in self.mask_fg if f.endswith('png')]
            MFs = [self.get_image(os.path.join(self.root_dir, 'mask_fg', f)) for f in self.mask_fg if f.endswith('jpg')]
        else:
            Ms = [self.get_image(os.path.join(self.root_dir, 'mask_bg', f), mask=True) for f in self.mask_bg if f.endswith('png')]
            MFs = [self.get_image(os.path.join(self.root_dir, 'mask_bg', f)) for f in self.mask_bg if f.endswith('jpg')]

        MFs = np.array(MFs).transpose((3, 0, 1, 2))
        Ms = np.array(Ms).transpose((1, 0, 2, 3))

        num_objects = torch.LongTensor([int(1)])
        info = {"point_id": point_output_id, "num_frames": len(self.frame_names)}

        # Fs = torch.from_numpy(Fs).unsqueeze(0).cuda()
        # MFs = torch.from_numpy(MFs).unsqueeze(0).cuda()
        # Ms = torch.from_numpy(Ms).unsqueeze(0).cuda()

        return MFs, Ms, num_objects, info


    def get_next_frame(self):
        frame = self.get_image(os.path.join(self.root_dir, self.point_dir[self.point_id], self.frame_names[self.frame_id])) # H x W x 3
        frame_input = np.array(frame).transpose((2, 0, 1)) # 3 x H x W
        frame_output_id = self.frame_output_ids[self.frame_id]

        self.frame_id += 1
        if self.frame_id == len(self.frame_names):
            self.point_id += 1

        return frame_input, frame_output_id 


    def get_image(self, f, mask=False):
        im = Image.open(f)
        if mask:
            im = (np.array(im.convert("L")) == 255).astype(np.uint8)
            # im = Image.fromarray(im)
            # im = np.array(im.convert("P")).astype(np.uint8)
            if self.infer_mask_fg:
                im = im[self.fg_cutline:]
            else:
                im = im[:self.bg_cutline]
            im = np.array([im == k for k in range(self.K)])
            return im.astype(np.uint8)

        # Normalize to [0, 1]
        else:
            im = np.array(im.convert("RGB"))
            if self.infer_mask_fg:
                im = im[self.fg_cutline:]
            else:
                im = im[:self.bg_cutline]
            return (im / 255).astype(np.float32)


class TennisDataset(object):
    """
    Generate player masks for Vid2player, load frames from video
    """

    def __init__(self, root_dir, video_path, point_bounds, infer_mask_fg):
        self.root_dir = root_dir
        self.K = 2

        # Video
        self.video_cap = cv2.VideoCapture(video_path)

        # Hack: crop image to reduce computation
        self.fg_cutline = 300
        self.bg_cutline = 500
        self.mask_filling_fg = np.zeros((self.fg_cutline, 1920), dtype=np.uint8)
        self.mask_filling_bg = np.zeros((1080 - self.bg_cutline, 1920), dtype=np.uint8)

        # Masks
        self.infer_mask_fg = infer_mask_fg
        if infer_mask_fg:
            mask_names = sorted([os.path.join(self.root_dir, 'mask_fg', f) for f in os.listdir(os.path.join(root_dir, 'mask_fg'))])
            Ms = [self.get_image(f, mask=True) for f in mask_names if f.endswith('png')]
            MFs = [self.get_image(f) for f in mask_names if f.endswith('jpg')]
        else:
            mask_names = sorted([os.path.join(self.root_dir, 'mask_bg', f) for f in os.listdir(os.path.join(root_dir, 'mask_bg'))])
            Ms = [self.get_image(f, mask=True) for f in mask_names if f.endswith('png')]
            MFs = [self.get_image(f) for f in mask_names if f.endswith('jpg')]

        self.MFs = np.array(MFs).transpose((3, 0, 1, 2))
        self.Ms = np.array(Ms).transpose((1, 0, 2, 3))

        self.point_bounds = point_bounds
        self.num_points = len(point_bounds)
        self.point_id = 0
        print("Run STM on video {}".format(video_path))
        print("Generating {} masks for {} points".format('fg' if infer_mask_fg else 'bg', len(point_bounds)))


    def get_next_point(self):
        if self.point_id == self.num_points:
            self.point_id = 0
            return -1

        self.video_cap.set(1, self.point_bounds[self.point_id][0])
        self.frame_id = 0

        print("Running {}th point, point bounds in {}".format(self.point_id, self.point_bounds[self.point_id]))

        num_objects = torch.LongTensor([int(1)])
        self.num_frames = self.point_bounds[self.point_id][1] - self.point_bounds[self.point_id][0] + 1

        return self.MFs, self.Ms, num_objects, self.num_frames


    def get_next_frame(self):
        ret, frame = self.video_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.infer_mask_fg:
            frame = frame[self.fg_cutline:]
        else:
            frame = frame[:self.bg_cutline]
        frame_input = (frame / 255).astype(np.float32).transpose((2, 0, 1)) # 3 x H x W

        frame_output_id = self.point_bounds[self.point_id][0] + self.frame_id

        self.frame_id += 1
        if self.frame_id == self.num_frames:
            self.point_id += 1

        return frame_input, frame_output_id 


    def get_image(self, f, mask=False):
        im = Image.open(f)
        if mask:
            im = (np.array(im.convert("L")) == 255).astype(np.uint8)
            # im = Image.fromarray(im)
            # im = np.array(im.convert("P")).astype(np.uint8)
            if self.infer_mask_fg:
                im = im[self.fg_cutline:]
            else:
                im = im[:self.bg_cutline]
            im = np.array([im == k for k in range(self.K)])
            return im.astype(np.uint8)

        # Normalize to [0, 1]
        else:
            im = np.array(im.convert("RGB"))
            if self.infer_mask_fg:
                im = im[self.fg_cutline:]
            else:
                im = im[:self.bg_cutline]
            return (im / 255).astype(np.float32)


if __name__ == '__main__':
    pass
