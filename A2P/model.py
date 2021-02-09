import os
import plot
import torch
import random
import numpy as np
from networks.net import Generator,Discriminator,AutoVCEncoder
from speaker_embedding import load_embd
from collections import OrderedDict

class Audio2GestureModel(object):
    def __init__(self,opt):
        self.opt = opt
        self.gen_checkpoint_path = opt.ckp_save_path+'_generator.pt'
        self.dis_checkpoint_path = opt.ckp_save_path+'_discriminator.pt'
        if os.path.exists(opt.ckp_save_path):
            os.makedir(opt.ckp_save_path)
        self.generator_network = Generator(opt.device,opt.num_class,opt.dropout)
        self.generator_optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_network.parameters()), lr=opt.lr_g,betas=(0.5, 0.999))
        # generator_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.generator_optimizer_ft, step_size=30, gamma=0.1)

        self.generator_network.to(opt.device)

        self.discriminator = Discriminator(opt.device,opt.num_class,opt.size_sample)

        if opt.adam:
            self.discriminator_optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=opt.lr_d,betas=(0.5, 0.999))
        else:
            self.discriminator_optimizer_ft = torch.optim.SGD(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=opt.lr_d)
        discriminator_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.discriminator_optimizer_ft, step_size=30, gamma=0.1)

        self.discriminator.to(opt.device)

        generator_params,discriminator_params = sum(p.numel() for p in self.generator_network.parameters() if p.requires_grad ),sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)

        ######INIT WEIGHTS#######
        self.init_weights(self.generator_network)
        self.init_weights(self.discriminator)
        #########################
        self.load_autovc()
        self.speaker_emb = load_embd(self.opt.batch_size)
        self.speaker_emb = self.speaker_emb.to(self.opt.device)
        
        if opt.mse:
            self.bce_loss = torch.nn.MSELoss().to(opt.device)
        else:
            self.bce_loss = torch.nn.BCELoss().to(opt.device)

    def init_weights(self,model):
        for param in model.parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass


    def loss_l2(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)


    def loss_l1(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target, size_average=None, reduce=None, reduction='mean')

    def set_input(self,data):
        self.audio = data['audio'].to(self.opt.device)
        self.pose = data['pose'].to(self.opt.device)

    def optimize_parameters(self):
        self.generator_network.train()
        self.discriminator.train()
        flip = random.random() > self.opt.flip
        valid = torch.Tensor(np.random.uniform(low=0.7, high=1.2, size=(len(self.pose)))).to(self.opt.device)
        fake = torch.Tensor(np.random.uniform(low=0.0, high=0.3, size=(len(self.pose)))).to(self.opt.device)
        with torch.no_grad():
            code = self.autovc(self.audio,self.speaker_emb)
        
        self.generator_optimizer_ft.zero_grad()
        fake_pose = self.generator_network(code,self.speaker_emb)
        fake_pose_for_d = fake_pose.detach()
        pred_fake = self.discriminator(fake_pose,self.speaker_emb)
        self.l1 = self.loss_l1(fake_pose,self.pose)
        self.l2 = self.loss_l2(fake_pose, self.pose)
        self.lg = self.bce_loss(torch.flatten(pred_fake),valid)
        self.lgen = self.opt.lambda_l1*self.l1+self.opt.lambda_discriminator*self.lg+self.opt.lambda_l2*self.l2
        self.lgen.backward()
        self.generator_optimizer_ft.step()

        self.discriminator_optimizer_ft.zero_grad()
        if flip:
            discriminator_pred_fake = self.discriminator(self.pose,self.speaker_emb)
            discriminator_pred_real = self.discriminator(fake_pose_for_d,self.speaker_emb)
        else:
            discriminator_pred_fake = self.discriminator(fake_pose_for_d,self.speaker_emb)
            discriminator_pred_real = self.discriminator(self.pose,self.speaker_emb)
        ld_real = self.bce_loss(torch.flatten(discriminator_pred_real),valid)
        ld_fake = self.bce_loss(torch.flatten(discriminator_pred_fake),fake)
        self.ld = (ld_real+ld_fake)*0.5
        self.ld.backward()
        self.discriminator_optimizer_ft.step()

    def show_loss(self):
        print("l1={},l2={},lg={},lgen={},ld={}".format(self.l1.item(),self.l2.item(),self.lg.item(),self.lgen.item(),self.ld.item()))

    def test(self):
        with torch.no_grad():
            self.generator_network.eval()
            code = self.autovc(self.audio,self.speaker_emb)
            fake_pose = self.generator_network(code,self.speaker_emb)
        fake_pose = fake_pose.cpu().numpy()
        real_pose = self.pose.cpu().numpy()
        fake_pose = plot.get_original_pose(fake_pose)
        real_pose = plot.get_original_pose(real_pose)
        plot.make_video(fake_pose,real_pose)

    def load_autovc(self):
        self.autovc = AutoVCEncoder(32,256,32)
        checkpoint = torch.load(self.opt.ckp_save_path+'autovc.ckpt',map_location=self.opt.device)
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model'].items():
            if 'encoder' in key:
                new_key = key[8:]
                new_state_dict[new_key] = val
        self.autovc.load_state_dict(new_state_dict)
        self.autovc.to(self.opt.device)
