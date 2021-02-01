import os
from data.animation_dataset import AnimationDataset
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random


class Gesture2ImageDataset(AnimationDataset):
  @staticmethod
  def modify_commandline_options(parser, is_train=True):
    parser = AnimationDataset.modify_commandline_options(parser, is_train)
    return parser

  def initialize(self,opt):
    self.opt = opt
    self.opt.img_size = (360,640,3)
    self.opt.length = 8
    self.data = pd.read_csv(os.path.join(opt.dataroot,"{}.csv".format(opt.phase)))

    transform_list=[]
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.CenterCrop(360))
    transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
    self.trans = transforms.Compose(transform_list)


  def __getitem__(self, index):
    row = np.load(self.data.iloc[index]['pose_fn'])
    pose = row['pose']
    imgs_paths = row['imgs']
    sample_index = random.sample(range(0,len(imgs_paths)),self.opt.length)
    ref_index = random.randint(0,len(imgs_paths)-1)
    pose_images = self.load_pose_images(pose,sample_index)
    images = self.load_images(imgs_paths,sample_index)
    ref_pose_image = self.load_ref_pose_image(pose,ref_index)
    ref_image = self.load_ref_image(imgs_paths,ref_index)
    return {'poses':pose_images,'images':images,'ref_pose':ref_pose_image,'ref_image':ref_image}

  def load_images(self,paths,sampleindex):
    imgs = []
    for idx in sampleindex:
      img = Image.open(paths[idx])
      img = self.trans(img)
      imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs

  def load_pose_images(self,pose,sampleindex):
    _,_,length = pose.shape
    pose_images = []
    for b in sampleindex:
      lab = np.zeros(self.opt.img_size).astype(np.uint8)
      for l in range(length):
        pos = pose[b,:,l]
        lab = cv2.circle(lab,(int(pos[0]),int(pos[1])),1,(255,255,255),-1)
      lab = Image.fromarray(lab).convert('RGB')
      lab = self.trans(lab)
      pose_images.append(lab)
    pose_images = torch.stack(pose_images)
    return pose_images
  
  def load_ref_pose_image(self,pose,ref_index):
    _,_,length = pose.shape
    lab = np.zeros(self.opt.img_size).astype(np.uint8)
    for l in range(length):
      pos = pose[ref_index,:,l]
      lab = cv2.circle(lab,(int(pos[0]),int(pos[1])),1,(255,255,255),-1)
    lab = Image.fromarray(lab).convert('RGB')
    lab = self.trans(lab)
    return lab

  def load_ref_image(self,paths,ref_index):
    img = Image.open(paths[ref_index])
    img = self.trans(img)
    return img
    
  def name(self):
    return 'Gesture2ImageDataset'

  def __len__(self):
    return len(self.data)