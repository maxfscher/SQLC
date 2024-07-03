import os
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,models,transforms
from PIL import Image
torch.manual_seed(42)
torch.backends.cudnn.deterministic=True
import time
import compressai
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models,models
import CLIC_Approaches.EncodeDecodeFunctions as EDF
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from NeuralImageEncoder import CompressAINetwork_ShapePreserving



def postprocessSQLC(feats):
    StainModel=CompressAINetwork_ShapePreserving()
    states=torch.load('/checkpoint_best_loss.pth.tar')
    StainModel.load_state_dict(states)
    decoded = StainModel.decode(feats['x_hat'])
    return decoded[0,0:3,:,:]/255

def DecodeCAI():
    models_path='Path'
    _,Runs,_=next(os.walk(models_path))
    lmds=[float(x.split('_')[-1]) for x in Runs]

    for lambda_train in lmds:
        if lambda_train <= 0.05:
            quality = 1
        else:
            quality = 8

        start=time.time()
        quality=quality
        model_function = models['bmshj2018-factorized'] 
        net = model_function(quality=quality)

        model_path = os.path.join(models_path, str(lambda_train) + 'checkpoint_best_loss.pth.tar')
        states = torch.load(model_path)

        try:
            net.load_state_dict(states['state_dict'])
        except Exception as e:
            print(e)

        net.update()
        net.eval()
        net=net.cuda()
        compressed_files = '/path/to/files/'
        _, _, files = next(os.walk(compressed_files))
        storePath='Out/Path'
        os.makedirs(storePath,exist_ok=True)
        for file in files:
            path = os.path.join(compressed_files, file)
            try:
                decoded, out = EDF.decode(path, net)
            except Exception as e:
                print(e)
                print(file)
                continue
            #print(file)
            """
            savename = file.replace('.bin','.png')
            x, y =(224,224)# map(int, coords.split('_'))
            outtransform = transforms.Resize((y, x))
            reconstruction = outtransform(decoded['img'])
            reconstruction.save(storePath +'/'+ savename)
            """
        end=time.time()
        print('quality factor',quality,'Time:',end-start)

DecodeCAI()
