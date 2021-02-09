class Option(object):
    def __init__(self):
        self.phase = 'train'
        self.dataroot = '/content/Gestures'
        self.pose_mode = 'no'
        self.batch_size = 16
        self.shuffle = True if self.phase == 'train' else False
        self.drop_last = True if self.phase == 'train' else False
        self.num_workers = 8
        self.start_epoch = 1
        self.train_epochs_num = 100
        self.ckp_save_path = ''
        self.adam = False
        self.num_class = 3
        self.lr_g = 0.002
        self.lr_d = 0.0002
        self.size_sample = 64
        self.workers = 12
        self.l1 = False
        self.mse = False
        self.flip = 1
        self.dropout = 0.5
        self.lambda_l1 = 100
        self.lambda_l2 = 100
        self.lambda_discriminator = 1
