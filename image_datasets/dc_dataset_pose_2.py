# coding=utf-8
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import json
import os
import os.path as osp
import random
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import PIL
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor
import random
import re


debug_mode=False

def tensor_to_image(tensor, image_path):
    """
    Convert a torch tensor to an image file.

    Args:
    - tensor (torch.Tensor): the input tensor. Shape (C, H, W).
    - image_path (str): path where the image should be saved.

    Returns:
    - None
    """
    if debug_mode: 
        # Check the tensor dimensions. If it's a batch, take the first image
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Check for possible normalization and bring the tensor to 0-1 range if necessary
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # Convert tensor to PIL Image
        to_pil = ToPILImage()
        img = to_pil(tensor)

        # Save the PIL Image
        dir_path = os.path.dirname(image_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img.save(image_path)


def directional_dilate(mask_1chw: torch.Tensor,
                       px: Union[int, Tuple[int,int,int,int]]) -> torch.Tensor:
    if isinstance(px, int):
        t = b = l = r = max(0, int(px))
    else:
        t, b, l, r = [max(0, int(v)) for v in px]

    m = (mask_1chw > 0.5).to(mask_1chw.dtype).unsqueeze(0)  # [1,1,H,W]

    if t > 0 or b > 0:
        m = F.pad(m, (0, 0, t, b), value=0)                 # (left,right,top,bottom)
        m = F.max_pool2d(m, kernel_size=(t + b + 1, 1), stride=1)

    if l > 0 or r > 0:
        m = F.pad(m, (l, r, 0, 0), value=0)
        m = F.max_pool2d(m, kernel_size=(1, l + r + 1), stride=1)

    m = (m > 0).to(mask_1chw.dtype)
    return m.squeeze(0)

class DressCodeTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        data_list: Optional[str] = None,
        flow_gt_dir=None
    ):
        super(DressCodeTestDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        # This code defines a transformation pipeline for image processing
        self.transform = transforms.Compose(
            [
                # Convert the input image to a PyTorch tensor
                transforms.ToTensor(),
                # Normalize the tensor values to a range of [-1, 1]
                # The first [0.5] is the mean, and the second [0.5] is the standard deviation
                # This normalization is applied to each color channel
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()

        self.order = order
        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        categories = []
        dataroot_names = []


        filename = os.path.join(dataroot_path, data_list)

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    bc = re.sub(r'\t',' ',line).split(" ")
                    if len(bc) != 3 : 
                        print(bc)
                        breakpoint()
                    im_name, _, category = re.sub(r'\t',' ',line).split(" ") #line.strip().split(" ")
                    c_name = im_name
                else:
                    if order == "paired":
                        bc = re.sub(r'\t',' ',line).split(" ")
                        if len(bc) < 3 : 
                            print(bc)
                            breakpoint()    
                        im_name, _, category = re.sub(r'\t',' ',line).split(" ") #line.strip().split(" ")
                        c_name = im_name
                    else:
                        bc = re.sub(r'\t',' ',line).split(" ")
                        if len(bc) < 3 : 
                            print(bc)
                            breakpoint()
                        im_name, c_name, category = re.sub(r'\t',' ',line).split(" ") #line.strip().split("    ")

                im_names.append(im_name)
                c_names.append(c_name)
                categories.append(category.strip('\n'))
                dataroot_names.append(dataroot_path)
        
        category_dict = {"0" : "upper_body", "1" : "lower_body", "2" : "dresses" }
        categories = [category_dict[c] for c in categories]
        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.categories = categories

        self.phase=phase
        self.flow_gt_dir=flow_gt_dir

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        category = self.categories[index]
        result = {}
        
        cloth = Image.open(os.path.join(self.dataroot, category, "images", c_name)).resize((self.width,self.height))
        cloth_pure = self.transform(cloth)
        # cloth_mask = Image.open(os.path.join(self.dataroot, self.phase, "cloth-mask", c_name)).resize((self.width,self.height))
        #cloth_mask = self.transform(cloth_mask)
        
        im_pil_big = Image.open(os.path.join(self.dataroot, category, "images", im_name)).resize((self.width,self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, category, "agnostic-mask", f"{im_name.split('_')[0]}.png")).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]

        mask = 1-mask
        im_mask = image * mask

        if self.phase == "test" :
            agnostic_mask = Image.open(os.path.join(self.dataroot, category, "agnostic-mask", f"{im_name.split('_')[0]}.png")).resize((self.width//16,self.height//16),resample=Image.NEAREST)
            agnostic_mask = self.toTensor(agnostic_mask)
            agnostic_mask = agnostic_mask[:1]
            result["agnostic-mask"] = agnostic_mask
            
            # cloth_mask = Image.open(os.path.join(self.dataroot, self.phase, "cloth-mask", c_name)).convert("L").resize((self.width//16,self.height//16),resample=Image.NEAREST)
            # cloth_mask = self.toTensor(cloth_mask)
            # cloth_mask = cloth_mask[:1]
            # result["cloth-mask"] = cloth_mask
 
        pose_img = Image.open(
            os.path.join(self.dataroot, category, "image-densepose-agnostic", f"{im_name.split('_')[0]}_4.jpg")
        ).resize((self.width,self.height))
        pose_img = self.transform(pose_img)  # [-1,1]
        result["pose_image"] = pose_img
 
        result["c_name"] = c_name
        result["im_name"] = im_name
        result["cloth_pure"] = cloth_pure
        #result["cloth_mask"] = cloth_mask
        
        # Concatenate image and garment along width dimension
        densepose_mask = torch.zeros_like(pose_img)
        inpaint_image = torch.cat([cloth_pure, im_mask, densepose_mask], dim=2)  # dim=2 is width dimension
        result["im_mask"] = inpaint_image
        
        GT_image = torch.cat([cloth_pure, image], dim=2)  # dim=2 is width dimension
        result["image"] = GT_image
        
        # Create extended black mask for garment portion
        garment_mask = torch.zeros_like(1-mask)  # Create mask of same size as original
        densepose_mask = torch.zeros_like(1-mask)
        extended_mask = torch.cat([garment_mask, 1-mask, densepose_mask], dim=2)  # Concatenate masks
        result["inpaint_mask"] = extended_mask
        result["category"] = category

        return result

    def __len__(self):
        return len(self.im_names)


if __name__ == "__main__":
    dataset = CPDataset("/data/user/gjh/VITON-HD", 512, mode="train", unpaired=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for data in loader:
        pass