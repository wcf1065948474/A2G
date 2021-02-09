import option
import dataset
import model
import torch

opt = option.Option()
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a2g_model = model.Audio2GestureModel(opt)
dataloader = dataset.Audio2GestureDataset(opt)
dataloader = torch.utils.data.DataLoader(dataloader,batch_size=opt.batch_size,shuffle=opt.shuffle,num_workers=opt.num_workers,drop_last=opt.drop_last)

for e in range(opt.start_epoch,opt.start_epoch+opt.train_epochs_num):
    for data in dataloader:
        a2g_model.set_input(data)
        a2g_model.optimize_parameters()

    print('e={}'.format(e))
    a2g_model.show_loss()
    a2g_model.test()

