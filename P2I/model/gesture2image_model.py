import torch
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import task, util
import itertools
import numpy as np
import random
import os, ntpath
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import cv2
from PIL import Image



class Gesture2Image(BaseModel):
    def name(self):
        return "Gesture2Image Animation"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--attn_layer', action=util.StoreList, metavar="VAL1,VAL2...")
        parser.add_argument('--kernel_size', action=util.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...")

        parser.add_argument('--layers', type=int, default=3, help='number of layers in G')
        parser.add_argument('--netG', type=str, default='gesture2image', help='The name of net Generator')
        parser.add_argument('--netD', type=str, default='res', help='The name of net Discriminator')
        parser.add_argument('--netD_V', type=str, default='temporal', help='The name of net Discriminator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        parser.add_argument('--ratio_g2d', type=float, default=0.1, help='learning rate ratio G to D')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for image reconstruction loss')
        parser.add_argument('--lambda_g', type=float, default=2.0, help='weight for generation loss')
        parser.add_argument('--lambda_correct', type=float, default=5.0, help='weight for generation loss')
        parser.add_argument('--lambda_style', type=float, default=500.0, help='weight for generation loss')
        parser.add_argument('--lambda_content', type=float, default=0.5, help='weight for generation loss')
        parser.add_argument('--lambda_regularization', type=float, default=0.0025, help='weight for generation loss')
        parser.add_argument('--frames_D_V', type=int, default=6, help='number of frames of D_V')

        parser.add_argument('--use_spect_g', action='store_false')
        parser.add_argument('--use_spect_d', action='store_false')
        parser.add_argument('--write_ext', type=str, help='png | jpg')
        
        parser.set_defaults(use_spect_g=False)
        parser.set_defaults(use_spect_d=True)

        # display
        parser.set_defaults(display_freq=50)
        parser.set_defaults(eval_iters_freq=500)
        parser.set_defaults(print_freq=50)
        parser.set_defaults(save_latest_freq=1000)
        parser.set_defaults(save_iters_freq=1000)

        parser.set_defaults(write_ext='png')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['app_gen','correctness_p', 'correctness_r',
                           'content_gen','style_gen',
                           'regularization_p','regularization_r',
                           'ad_gen', 'dis_img_gen',
                           'ad_gen_v', 'dis_img_gen_v']

        self.visual_names = ['ref_image', 'ref_pose', 'poses', 'images','img_gen']        
        self.model_names = ['G', 'D', 'D_V']

        self.net_G = network.define_g(opt, filename='generator', image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64, img_f=512,
                                      layers=opt.layers, num_blocks=2, use_spect=opt.use_spect_g, attn_layer=opt.attn_layer, 
                                      norm='instance', activation='LeakyReLU', extractor_kz=opt.kernel_size)

        # self.flow2color = util.flow2color()
        # self.convert2skeleton = openpose_utils.tensor2skeleton(spatial_draw=True)

        self.net_D = network.define_d(opt, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)
        self.net_D_V = network.define_d(opt, name=opt.netD_V, input_length=opt.frames_D_V, ndf=32, img_f=128, layers=4, use_spect=opt.use_spect_d)

        if self.isTrain:
            self.GANloss = external_function.AdversarialLoss(opt.gan_mode).to(opt.device)
            self.L1loss = torch.nn.L1Loss()
            self.Correctness = external_function.PerceptualCorrectness().to(opt.device)

            self.Regularization = external_function.MultiAffineRegularizationLoss(kz_dic=opt.kernel_size).to(opt.device)
            self.Vggloss = external_function.VGGLoss().to(opt.device)

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = torch.optim.Adam(itertools.chain(
                                filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                filter(lambda p: p.requires_grad, self.net_D_V.parameters())),
                                lr=opt.lr*opt.ratio_g2d, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_D)

        else:
            self.results_dir_base = self.opt.results_dir
        self.esp=1e-6            
        self.setup(opt)


    def set_input(self, data):
        self.poses = data['poses'].cuda()
        self.images = data['images'].cuda()

        self.ref_pose = data['ref_pose'].cuda()
        self.ref_image = data['ref_image'].cuda()
        self.pre_pose = self.ref_pose
        self.pre_image = self.ref_image

    def optimize_parameters(self):
        self.P_gt_previous_recoder = torch.cat([self.ref_image.unsqueeze(1),self.images[:,:-1,...]],1)
        self.img_gen, self.flow_fields, \
        self.occlusion = self.net_G(self.poses, 
                                    self.ref_image, 
                                    self.ref_pose)

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def if_cross_eval(self,image_paths, ref_paths):
        video_driven = image_paths[0][0].split('/')[-2]
        video_ref = ref_paths[0].split('/')[-2]
        return False if video_ref == video_driven else True

    def write2video(self, name_list):
        images=[]
        for name in name_list:
            images.append(sorted(glob.glob(self.opt.results_dir+'/*_'+name+'.'+self.opt.write_ext)))

        image_array=[]
        for i in range(len(images[0])):
            cat_im=None
            for image_list in images:
                im = cv2.imread(image_list[i])
                if cat_im is not None:
                    cat_im = np.concatenate((cat_im, im), axis=1)
                else:
                    cat_im = im
            image_array.append(cat_im) 

        res=''
        for name in name_list:
            res += (name +'_')
        out_name = self.opt.results_dir+'_'+res+'.mp4' 
        print('write video %s'%out_name)  
        height, width, layers = cat_im.shape
        size = (width,height)
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
 
        for i in range(len(image_array)):
            out.write(image_array[i])
        out.release()


    def get_current_visuals(self):
        imgs = []
        for name in self.visual_names:
            img = getattr(self,name)
            if name == 'poses' or name == 'images':
                img = img[:,0,...]
            if name == 'img_gen':
                img = img[0].detach()
            imgs.append(util.tensor2im(img))
        imgs = np.concatenate(imgs,1)
        imgs = Image.fromarray(imgs)
        imgs.save('/content/Global-Flow-Local-Attention/result/res.jpg')
        # """Return visualization images"""
        # visual_ret = OrderedDict()
        # for name in self.visual_names:
        #     if isinstance(name, str):
        #         value = getattr(self, name)
        #         if '_step' in name:
        #             value = value[0]
        #             list_value=[]
        #             for i in range(value.size(0)):
        #                 _value = value[i, -3:,...]
        #                 list_value.append(_value.unsqueeze(0))
        #             value=list_value
        #         if '_skeleton' in name:
        #             value = value[0, -3:,...]
        #         if 'flow_field' in name or 'occlusion' in name:
        #             list_value = [item for sub_list in value for item in sub_list]
        #             value = list_value

        #         if isinstance(value, list):
        #             # visual multi-scale ouputs
        #             for i in range(len(value)):
        #                 visual_ret[name + str(i)] = self.convert2im(value[i], name)
        #             # visual_ret[name] = util.tensor2im(value[-1].data)       
        #         else:
        #             visual_ret[name] = self.convert2im(value, name)         
        # return visual_ret 


    def test(self, save_features=False, save_all=False, generate_edge=True):
        """Forward function used in test time"""
        self.img_gen, flow_fields, \
        occlusion, P_previous_recoder \
                        = self.net_G(self.gen_skeletons, 
                                    self.ref_image, 
                                    self.ref_skeleton,
                                    self.pre_image,
                                    self.pre_skeleton)        
        self.pre_image = self.img_gen[-1]
        self.pre_skeleton = self.gen_skeletons[:,-1,...]

        self.image_paths = self.check_image_paths()
        if self.frame_idx == self.opt.start_frame+self.opt.n_frames_pre_load_test:
            self.save_results(self.ref_image, data_name='ref', data_ext=self.opt.write_ext)

        # save the generated image
        gen_image = torch.cat(self.img_gen, 0)
        self.save_results(gen_image, data_name='vis', data_ext=self.opt.write_ext)

        # save the gt image
        gt_image = self.gen_images.squeeze(0)
        self.save_results(gt_image, data_name='gt', data_ext=self.opt.write_ext)     

        # save the skeleton image
        current_skeleton = self.convert2skeleton(self.openpose_kp[0,...], 'COCO_17')
        obtained_skeleton = self.convert2skeleton(self.video2d_kp[0], 'human36m_17')
        for i in range(len(current_skeleton)):
            short_path = ntpath.basename(self.image_paths[i])  
            name = os.path.splitext(short_path)[0]
            util.mkdir(self.opt.results_dir)

            in_img_name = '%s_%s.%s' % (name, 'skeleton_in', self.opt.write_ext)
            img_path = os.path.join(self.opt.results_dir, in_img_name)
            skeleton_in = current_skeleton[i].astype(np.uint8)       
            util.save_image(skeleton_in, img_path) 

            out_img_name = '%s_%s.%s' % (name, 'skeleton_out', self.opt.write_ext)
            img_path = os.path.join(self.opt.results_dir, out_img_name)
            skeleton_out = obtained_skeleton[i].astype(np.uint8)       
            util.save_image(skeleton_out, img_path) 

        if self.change_seq:
            name_list=['gt', 'vis', 'skeleton_in', 'skeleton_out']
            self.write2video(name_list)            

    def check_image_paths(self):
        names = []
        for name in self.image_paths:
            if isinstance(name, tuple):
                name = name[0]
            names.append(name)
        return names

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5

        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty
        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        
        # Spatial GAN Loss
        base_function._unfreeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        real = self.images[:,i,...]
        self.loss_dis_img_gen = self.backward_D_basic(self.net_D, real, fake)

        # Temporal GAN Loss
        base_function._unfreeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen)-self.opt.frames_D_V+1)
        fake = []
        real = []        
        for frame in range(self.opt.frames_D_V):
            fake.append(self.img_gen[i+frame].unsqueeze(2))
            real.append(self.images[:,i+frame,...].unsqueeze(2))
        fake = torch.cat(fake, dim=2)
        real = torch.cat(real, dim=2)
        self.loss_dis_img_gen_v = self.backward_D_basic(self.net_D_V, real, fake)

    def backward_G(self):
        """Calculate training loss for the generator"""
        loss_style_gen, loss_content_gen, loss_app_gen, loss_pose=0,0,0,0

        # Calculate the Reconstruction Loss
        for i in range(len(self.img_gen)):
            gen = self.img_gen[i]
            gt = self.images[:,i,...]
            loss_app_gen += self.L1loss(gen, gt)

            content_gen, style_gen = self.Vggloss(gen, gt) 
            loss_style_gen += style_gen
            loss_content_gen += content_gen

        self.loss_style_gen = loss_style_gen * self.opt.lambda_style
        self.loss_content_gen = loss_content_gen * self.opt.lambda_content            
        self.loss_app_gen = loss_app_gen * self.opt.lambda_rec


        loss_correctness_p, loss_regularization_p=0, 0
        loss_correctness_r, loss_regularization_r=0, 0

        # Calculate the Sampling Correctness Loss and Regularization Loss
        for i in range(len(self.flow_fields)):
            flow_field_i = self.flow_fields[i]
            flow_p, flow_r=[],[]
            for j in range(0, len(flow_field_i), 2):
                flow_p.append(flow_field_i[j])
                flow_r.append(flow_field_i[j+1])

            mask = None
            correctness_r = self.Correctness(self.images[:,i,...], self.ref_image, 
                                            flow_r, self.opt.attn_layer, mask)
            correctness_p = self.Correctness(self.images[:,i,...], self.P_gt_previous_recoder[:,i,...], 
                                            flow_p, self.opt.attn_layer, mask)
            loss_correctness_p += correctness_p
            loss_correctness_r += correctness_r

            loss_regularization_p += self.Regularization(flow_p)
            loss_regularization_r += self.Regularization(flow_r)


        self.loss_correctness_p = loss_correctness_p * self.opt.lambda_correct     
        self.loss_correctness_r = loss_correctness_r * self.opt.lambda_correct   
        self.loss_regularization_p = loss_regularization_p * self.opt.lambda_regularization
        self.loss_regularization_r = loss_regularization_r * self.opt.lambda_regularization


        # Spatial GAN Loss
        base_function._freeze(self.net_D)
        i = np.random.randint(len(self.img_gen))
        fake = self.img_gen[i]
        D_fake = self.net_D(fake)
        self.loss_ad_gen = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # Temporal GAN Loss        
        base_function._freeze(self.net_D_V)
        i = np.random.randint(len(self.img_gen)-self.opt.frames_D_V+1)
        fake = []
        for frame in range(self.opt.frames_D_V):
            fake.append(self.img_gen[i+frame].unsqueeze(2))
        fake = torch.cat(fake, dim=2)
        D_fake = self.net_D_V(fake)
        self.loss_ad_gen_v = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        total_loss = 0
        for name in self.loss_names:
            if name != 'dis_img_gen_v' and name != 'dis_img_gen':
                total_loss += getattr(self, "loss_" + name)
        total_loss.backward()