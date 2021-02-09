import torch
import torch.nn as nn
import torch.nn.functional as F
from .stgc import *
from .graph import Graph

class Generator(nn.Module):

    def __init__(self,device,num_class,dropout,train_phase=True,num_joints=49):
        super(Generator,self).__init__()

        ##############################
        ####GRAPHS INITIALIZATIONS####
        ##############################

        # cols1 = [0,2,5,28,31,35,39,43,47,7,10,14,18,22,26]
        # cols2 = [0,3,9]
        # cols3 = [0]
        
        # self.graph25 = Graph(49,[(0,1),(1,2),(2,3),(3,28),(28,29),(29,30),(30,31),(31,32),
        #                     (28,33),(33,34),(34,35),(35,36),(28,37),(37,38),(38,39),(39,40),
        #                     (28,41),(41,42),(42,43),(43,44),(28,45),(45,46),(46,47),(47,48),
        #                     (0,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),
        #                     (7,12),(12,13),(13,14),(14,15),(7,16),(16,17),(17,18),(18,19),
        #                     (7,20),(20,21),(21,22),(22,23),(7,24),(24,25),(25,26),(26,27)],1)

        cols1 = [0,5,7,2,8]
        cols2 = [0,3,4]
        cols3 = [0]
        
        self.graph25 = Graph(9,[(0,1),(1,2),(2,3),(3,8),
                            (0,4),(4,5),(5,6),(6,7)],1)

        self.ca25 = torch.tensor(self.graph25.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a25 = torch.tensor(self.graph25.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)
        _,l1 = self.graph25.getLowAjd(cols1)
        
        self.graph11 = Graph(5,l1,0)
        self.ca11 = torch.tensor(self.graph11.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a11 = torch.tensor(self.graph11.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)
        _,l2 = self.graph11.getLowAjd(cols2)

        self.graph3 = Graph(3,l2,0)
        self.ca3 = torch.tensor(self.graph3.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a3 = torch.tensor(self.graph3.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)
        _,l3 = self.graph3.getLowAjd(cols3)

        self.graph1 = Graph(1,l3,0)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)
        ##############################
        #############END##############
        ##############################
        self.num_class = num_class
        self.num_joints = num_joints
        self.device = device
        self.train_phase = train_phase

        self.embed = nn.Embedding(self.num_class,512)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
        self.act = self.lrelu


        self.norm1 = nn.BatchNorm2d(256)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(32)
        self.norm5 = nn.BatchNorm2d(16)

        ########STGCN#######
        self.gcn0 = st_gcn(320,320,(1,self.ca1.size(0)))
        self.gcn1 = st_gcn(320,256,(1,self.ca3.size(0)))
        self.gcn2 = st_gcn(256,128,(1,self.ca3.size(0)))
        self.gcn3 = st_gcn(128,64,(3,self.ca11.size(0)))
        self.gcn4 = st_gcn(64,32,(3,self.ca11.size(0)))
        self.gcn5 = st_gcn(32,16,(7,self.ca25.size(0)))
        self.gcn6 = st_gcn(16,2,(7,self.ca25.size(0)))
        #########END##########

        #######GRAPH-UPSAMPLING########
        self.ups1 = UpSampling(1,3,self.a3,320)
        self.ups2 = UpSampling(3,5,self.a11,256)
        self.ups3 = UpSampling(5,9,self.a25,64)
        ###############END##############

        #######TEMPORAL-UPSAMPLING########
        self.upt1 = nn.ConvTranspose2d(256,256,(2,1),stride=(2,1))
        self.upt2 = nn.ConvTranspose2d(128,128,(2,1),stride=(2,1))
        self.upt3 = nn.ConvTranspose2d(64,64,(2,1),stride=(2,1))
        self.upt4 = nn.ConvTranspose2d(32,32,(2,1),stride=(2,1))
        ###############END##############
        
    def forward(self,code,embd):
        #batch,channels,time,vertex
        ######CONDITIONING#########
        # if self.train_phase:
        #     emb = self.embed(y).view(len(z),512,1,1).repeat(1,1,z.shape[2],1)
        #     inp = torch.cat((z,emb),1)
        # else:
        #     ######TESTING CODE##########
        #     emb = self.embed(y).unsqueeze(2).repeat(1,1,4).permute(1,0,2).reshape(len(z),512,-1,1)
        #     inp = torch.cat((z[:,:,:emb.shape[2]],emb),1)
            ###########################
        ################################
        embd = embd[:,:,None,None].repeat(1,1,code.shape[2],1)
        inp = torch.cat((embd,code),1)

        aux = self.lrelu(self.gcn0(inp,self.ca1))
        aux = self.act(self.norm1(self.gcn1(self.ups1(aux),self.ca3)))
        aux = self.dropout(self.act(self.norm2(self.gcn2(self.upt1(aux),self.ca3))))
        aux = self.act(self.norm3(self.gcn3(self.ups2(self.upt2(aux)),self.ca11)))
        aux = self.dropout(self.act(self.norm4(self.gcn4(self.upt3(aux),self.ca11))))
        aux = self.act(self.norm5(self.gcn5(self.ups3(self.upt4(aux)),self.ca25)))
        
        aux = self.gcn6(aux,self.ca25)
        return aux
    
class Discriminator(nn.Module):

    def __init__(self,device,num_class,size_sample,num_joints=49):
        super(Discriminator,self).__init__()

        ##############################
        ####GRAPHS INITIALIZATIONS####
        ##############################

        # cols1 = [0,2,5,28,31,35,39,43,47,7,10,14,18,22,26]
        # cols2 = [0,3,9]
        # cols3 = [0]
        
        # self.graph25 = Graph(49,[(0,1),(1,2),(2,3),(3,28),(28,29),(29,30),(30,31),(31,32),
        #                     (28,33),(33,34),(34,35),(35,36),(28,37),(37,38),(38,39),(39,40),
        #                     (28,41),(41,42),(42,43),(43,44),(28,45),(45,46),(46,47),(47,48),
        #                     (0,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),
        #                     (7,12),(12,13),(13,14),(14,15),(7,16),(16,17),(17,18),(18,19),
        #                     (7,20),(20,21),(21,22),(22,23),(7,24),(24,25),(25,26),(26,27)],1)

        cols1 = [0,5,7,2,8]
        cols2 = [0,3,4]
        cols3 = [0]
        
        self.graph25 = Graph(9,[(0,1),(1,2),(2,3),(3,8),
                            (0,4),(4,5),(5,6),(6,7)],1)

        self.ca25 = torch.tensor(self.graph25.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a25 = torch.tensor(self.graph25.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)
        _,l1 = self.graph25.getLowAjd(cols1)
        
        self.graph11 = Graph(5,l1,0)
        self.ca11 = torch.tensor(self.graph11.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a11 = torch.tensor(self.graph11.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)
        _,l2 = self.graph11.getLowAjd(cols2)

        self.graph3 = Graph(3,l2,0)
        self.ca3 = torch.tensor(self.graph3.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a3 = torch.tensor(self.graph3.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)
        _,l3 = self.graph3.getLowAjd(cols3)

        self.graph1 = Graph(1,l3,0)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)

        ##############################
        #############END##############
        ##############################

        self.size_sample = size_sample
        self.num_joints = num_joints
        self.device = device
        self.num_class = num_class

        self.embed = nn.Embedding(self.num_class,self.num_joints)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout()

        self.act = self.lrelu


        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)

        ########STGCN#######
        self.gcn0 = st_gcn(258,2,(7,self.ca25.size(0)))
        self.gcn1 = st_gcn(2,32,(7,self.ca25.size(0)))
        self.gcn2 = st_gcn(32,64,(3,self.ca11.size(0)))
        self.gcn3 = st_gcn(64,128,(3,self.ca11.size(0)))
        self.gcn4 = st_gcn(128,256,(1,self.ca3.size(0)))
        self.gcn5 = st_gcn(256,1,(1,self.ca1.size(0)))
        #########END##########

        #######GRAPH-DOWNSAMPLING########
        self.dws1 = DownSampling(9,5,self.a25,64)
        self.dws2 = DownSampling(5,3,self.a11,256)
        self.dws3 = DownSampling(3,1,self.a3,1)
        ###############END##############

        #######TEMPORAL-DOWNSAMPLING########
        self.dwt1 = nn.Conv2d(32,32,(int(self.size_sample/2)+1,1))
        self.dwt2 = nn.Conv2d(64,64,(int(self.size_sample/4)+1,1))
        self.dwt3 = nn.Conv2d(128,128,(int(self.size_sample/8)+1,1))
        self.dwt4 = nn.Conv2d(256,256,(int(self.size_sample/16)+1,1))
        self.dwt5 = nn.Conv2d(1,1,(int(self.size_sample/16),3))      
        ###############END##############


    def forward(self,x,y):

        #################CONDITIONING################
        emb = y.view(len(x),256,1,1).repeat(1,1,self.size_sample,9)
        aux = torch.cat((x,emb),1)
        inp = self.lrelu(self.gcn0(aux,self.ca25))
        ############################################

        # pdb.set_trace()
        aux = self.lrelu(self.dwt1(self.gcn1(inp,self.ca25)))
        aux = self.lrelu(self.norm1(self.dws1(self.dwt2(self.gcn2(aux,self.ca25)))))
        aux = self.lrelu(self.norm2(self.dwt3(self.gcn3(aux,self.ca11))))
        aux = self.lrelu(self.norm3(self.dws2(self.dwt4(self.gcn4(aux,self.ca11)))))
        aux = self.dwt5(self.gcn5(aux,self.ca3))
    
        return self.sigmoid(aux)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class AutoVCEncoder(nn.Module):
    """AutoVCEncoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(AutoVCEncoder, self).__init__()
        self.dim_neck = dim_neck
        # self.freq = freq
        self.freq = 64
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1)-self.freq, self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        codes = torch.stack(codes, dim=1)
        codes = codes.unsqueeze(-1)
        codes = codes.permute(0,2,1,3)
        return codes