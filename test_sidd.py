import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
import glob
import torch.nn.init as init
import os, sys
import time
from datetime import datetime
from mydata import myDataset

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.__stdout__
        self.log = open(fileN, "a+")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.log.flush() 

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--outf", type=str, default="log_sidd", help='path of log files')
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
parser.add_argument("--model_path", type=str, default='./experiments/pre_trained.pth', help="path of pre-trained R2R model")
parser.add_argument("--val_path", type=str, default='./', help="path of SIDD val(test) data")
parser.add_argument("--phase", type=str, default='validation', help="validation or test for SIDD data")
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
import scipy.io as scio

def main():
    # Load dataset
    current_time = time.strftime('%Y%m%d-%H:%M:%s',time.localtime())
    MODEL_PATH = './experiments/sidd_dncnn/'+current_time
    os.makedirs(MODEL_PATH,exist_ok=True)
    sys.stdout = Logger(MODEL_PATH+'/train.txt')
    os.system('cp ./train_sidd_dncnn.py '+MODEL_PATH)

    print('Loading dataset ...\n')

    # dataset_train=myDataset(path=opt.r2r_path)
    # loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    
    # print("# of training samples: %d\n" % int(len(dataset_train)))
    net = DnCNN(image_channels=3, depth=20)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(opt.model_path))
    # training
    dtype = torch.cuda.FloatTensor
    now = datetime.now()
    print('Start training.....',now.strftime("%H:%M:%S"))
    model.eval()
    if opt.phase == 'validation':
        var_gt = scio.loadmat(os.path.join(opt.val_path,'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb'].astype(np.float32)/255.
        var_noisy = scio.loadmat(os.path.join(opt.val_path,'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb'].astype(np.float32)/255. 
        psnr_val = 0

        with torch.no_grad():
            for k in range(40):
                for m in range(32):
                    noisy = torch.from_numpy(var_noisy[k,m,:,:,::-1].copy().transpose(2,0,1)).unsqueeze(0).type(dtype)
                    # model.eval()
                    pred = model(noisy)[0].detach().cpu().numpy()
                    psnr_val = psnr_val+compare_psnr(var_gt[k,m,:,:,::-1].transpose(2,0,1),pred)
        print('validation PSNR: %f' %(psnr_val/40/32))               
        del var_gt,var_noisy                         
    elif opt.phase =='test':
        test_noisy = scio.loadmat(os.path.join(opt.val_path,'BenchmarkNoisyBlocksSrgb.mat'))['BenchmarkNoisyBlocksSrgb'].astype(np.float32)/255.
        with torch.no_grad():
            test = []
            for k in range(40):
                tem_test = []
                for m in range(32):
                    noisy = torch.from_numpy(test_noisy[k,m,:,:,::-1].copy().transpose(2,0,1)).unsqueeze(0).type(dtype)
                    pred = model(noisy)
                    pred = (np.clip((pred).detach().cpu().numpy(),0,1)*255).astype(np.uint8)
                    tem_test.append(pred)
                test.append(np.concatenate(tem_test,axis=0).transpose(0,2,3,1)[np.newaxis])
            test = np.concatenate(test,axis=0)
            from scipy.io import savemat
            test = {'DenoisedBlocksSrgb': test[:,:,:,:,::-1]}
            os.makedirs('./test_result',exist_ok=True)
            savemat("./test_result/R2R_sidd_test.mat", test)
            print('test of SIDD is done, result can be found at ./test_result/R2R_sidd_test.mat')         

if __name__ == "__main__":
    main()

