import sys
import time
import os
import csv
import torch
from util import Logger
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np


# CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --dataroot  {Test-dir} --model_path {Model-Path}
# CUDA_VISIBLE_DEVICES=0 python eval_test8gan.py --dataroot  /opt/data/private/tcc/data/data/CNNDetection/test/ --model_path ./model_epoch_Acc_92.0253457924526_epoch_21_steps99022.pth

vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(opt.model_path).replace('.pth', '')

dataroot = opt.dataroot
print(f'Dataroot {opt.dataroot}')
print(f'Model_path {opt.model_path}')

# get model
model = resnet50(num_classes=1)
from collections import OrderedDict
from copy import deepcopy
state_dict = torch.load(opt.model_path, map_location='cpu')['model']
net_params = sum(map(lambda x: x.numel(), model.parameters()))
print(f'Model parameters {net_params:,d}')
pretrained_dict = OrderedDict()
for ki in state_dict.keys():
    pretrained_dict[ki[7:]] = deepcopy(state_dict[ki])
model.load_state_dict(pretrained_dict, strict=True)
# model.load_state_dict(torch.load(opt.model_path, map_location='cpu')['model'])
model.cuda()
model.eval()

accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = False    # testing without resizing by default
    opt.no_crop = True    # testing without cropping by default
    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc);aps.append(ap)
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

