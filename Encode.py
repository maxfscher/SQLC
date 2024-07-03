import glob
import os
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,models,transforms
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(42)
torch.backends.cudnn.deterministic=True
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models,models
import CLIC_Approaches.EncodeDecodeFunctions as EDF
from NeuralImageEncoder import CompressAINetwork_ShapePreserving
import histomicstk as htk
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True
seed_everything(42)
def encodeUNet():
    model = FiveUNet_ExtractFuncs(BneckOut=512)
    states = '/home/m813r/Projects/NeuralImageCompression/CLICchallenge/checkpoint_best_loss.pth.tar'
    states = torch.load(states)
    model.load_state_dict(states)
    model.eval()
    path='/home/m813r/Downloads/clic2024_validation_image'
    _,_,files=next(os.walk(path))
    for file in files:
        name=file
        file=Image.open(os.path.join(path,file))
        shape=file.size
        file=np.asarray(file)
        file=transforms.ToTensor()(file)
        transformation=transforms.Resize((224,224))
        test=transformation(file)
        Reconstruction,latents=model(test.unsqueeze(0))
        name=name[:-4]
        file_name=name+'-'+str(shape[0])+'_'+str(shape[1])+'.pt'
        torch.save(latents,'/home/m813r/Projects/NeuralImageCompression/CLICchallenge/compressed_images/'+file_name)

def StainDeconvHistomics(PILImg):
    stains = ['hematoxylin',  # nuclei stain
              'eosin',  # cytoplasm stain
              'dab']  # set to null if input contains only two stains
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    img=np.asarray(PILImg)
    W = np.array([stain_color_map[st] for st in stains]).T
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(img, W)

    return np.concatenate((img,imDeconvolved.Stains),axis=2,dtype=np.float32)
def preprocessSQLC(img,model):
    StainModel=model
    transformations = transforms.Compose([transforms.CenterCrop(224), transforms.Lambda(StainDeconvHistomics), transforms.ToTensor()])
    img=transformations(img).unsqueeze(0).cuda()
    feats, recon = StainModel(img)
    return feats



def CompressCAImodels():
    models_path='/home/m813r/ClusterCheckpoints/NeuralImageCompression/TrainedCAEmodels/RMS/_bmshj2018-factorized/AdaptableBN_RandomAugment'
    _,Runs,_=next(os.walk(models_path))
    lmds=[float(x.split('_')[-1]) for x in Runs]
    #lmds = [0.001,0.003,0.004,0.5,0.8,1.0,0.3]
    for lambda_train in lmds:
        if lambda_train <= 0.05:
            quality = 1
        else:
            quality = 8

        #quality=lambda_train
        model_function = models['bmshj2018-factorized']  # models['bmshj2018-factorized']
        net = model_function(quality=quality)#,pretrained=True)

        model_path = os.path.join(models_path, str(lambda_train) + 'checkpoint_best_loss.pth.tar')
        states = torch.load(model_path)

        try:
            net.load_state_dict(states['state_dict'])
        except Exception as e:
            print(e)

        net.update()
        net.eval()
        net=net.cuda()
        test_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        path = '/home/m813r/Projects/NeuralImageCompression/BitRateEval/Homogenized_uncompressed_Data/ValidationData'
        _, _, files = next(os.walk(path))
        start=time.time()
        for file in files:
            name = file
            file = Image.open(os.path.join(path, file))
            shape = file.size
            file = np.asarray(file)
            file = transforms.ToTensor()(file)
            transformation = transforms.Resize((224, 224))
            test = transformation(file)
            test=test.unsqueeze(0).cuda()
            BaseOutPath='/home/m813r/Projects/NeuralImageCompression/BitRateEval/Homogenized_uncompressed_Data/RetrainedSPL2/'+str(lambda_train)+'/'
            os.makedirs(BaseOutPath,exist_ok=True)
            name = name[:-4]
            file_name = name  + '.bin'
            FileOutputPath=os.path.join(BaseOutPath,file_name)
            compressed, codec, out, x = EDF.encode(test, quality, net, FileOutputPath)
        end=time.time()
        print('quality factor:' ,quality)
        print('Time for encoding:',end-start)


"""
With the CompressAI models i somehow can not do writing to disk and visual reconstruction together.
Most likely due to issues with the entropy coding schemes
So I have at first a CompressSQLC function that does the compression as bin files on the disk
and then a function that does the decoding and visual reconstruction and storing visually the output 
"""
def CompressSQLC():
    models_path='/home/m813r/ClusterCheckpoints/NeuralImageCompression/TrainedCAEmodels_SQLC/ReproduceRuns03_oldStainModel_HopeSameResults_SaveStainModelWeights/_bmshj2018-factorized/AdaptableBN_RandomAugment/'
    _, Runs, _ = next(os.walk(models_path))
    lmds = [float(x.split('_')[-1]) for x in Runs]
    model_function = models['bmshj2018-factorized']
    for lm in lmds:
        if lm <= 0.05:
            quality = 1
        else:
            quality = 8
        net = model_function(quality=quality)
        states = torch.load(models_path + str(lm) + 'checkpoint_best_loss.pth.tar')
        try:
            net.load_state_dict(states['state_dict'])
        except Exception as e:
            print(e)
        net = net.cuda()
        net.eval()
        net.update()
        stainModel=CompressAINetwork_ShapePreserving()
        #for run in IndividuallyTrainedStainModel:
        states = torch.load(models_path+str(lm)+'StainModelcheckpoint_best_loss.pth.tar')
        stainModel.load_state_dict(states['state_dict'])
        stainModel.eval()
        stainModel=stainModel.cuda()
        def SQLCCompression(InputImage):
            InputImage = InputImage.unsqueeze(0)
            InputImage = InputImage.cuda()
            feats, recon = stainModel(InputImage)
            reconstructed = net(feats)
            decoded = stainModel.decode(reconstructed['x_hat'])
            decoded = decoded[0, 0:3, :, :] / 255
            return Image.fromarray(np.asarray((decoded.permute(1, 2, 0).detach().cpu()) * 255, dtype=np.uint8))

        OutPutBasePath = '/home/m813r/Projects/NeuralImageCompression/SegmentationBaseline/nnUNET/0/MeasureBPP/SQLC/BothCheckpoints'
        os.makedirs(OutPutBasePath, exist_ok=True)
        StainConv = 'IndividualRun'
        CompressionModel = 'Original'
        OutPath = os.path.join(OutPutBasePath, str(lm))
        os.makedirs(OutPath, exist_ok=True)


        path = '/home/m813r/Projects/NeuralImageCompression/SegmentationBaseline/nnUNET/0/imgs4compression_bad_imgs_excluded'
        _, _, files = next(os.walk(path))
        for file in files:
            name = file
            file = Image.open(os.path.join(path, file))
            test=preprocessSQLC(file,stainModel)
            test=test.cuda()
            name = name[:-4]
            file_name = name  + '.bin'
            FileOutputPath=os.path.join(OutPath,file_name)
            compressed, codec, out, x = EDF.encode(test, quality, net, FileOutputPath)





def DecodeSQLC():
    models_path='/home/m813r/ClusterCheckpoints/NeuralImageCompression/TrainedCAEmodels_SQLC/ReproduceRuns03_oldStainModel_HopeSameResults_SaveStainModelWeights/_bmshj2018-factorized/AdaptableBN_RandomAugment/'
    _, Runs, _ = next(os.walk(models_path))
    lmds = [float(x.split('_')[-1]) for x in Runs]
    model_function = models['bmshj2018-factorized']
    for lm in lmds:

        if lm <= 0.05:
            quality = 1
        else:
            quality = 8
        net = model_function(quality=quality)
        states = torch.load(models_path + str(lm) + 'checkpoint_best_loss.pth.tar')
        try:
            net.load_state_dict(states['state_dict'])
        except Exception as e:
            print(e)
        net = net.cuda()
        net.update()

        stainModel=CompressAINetwork_ShapePreserving()
        #for run in IndividuallyTrainedStainModel:
        states = torch.load(models_path+str(lm)+'StainModelcheckpoint_best_loss.pth.tar')
        stainModel.load_state_dict(states['state_dict'])
        #stainModel.update()
        stainModel.train()
        stainModel=stainModel.cuda()


        def SQLCCompression(InputImage):
            InputImage = InputImage.unsqueeze(0)
            InputImage = InputImage.cuda()
            with torch.no_grad():
                feats, recon = stainModel(InputImage)
            reconstructed = net(feats)
            with torch.no_grad():
                decoded = stainModel.decode(reconstructed['x_hat'])
            decoded = decoded[0, 0:3, :, :] / 255
            return Image.fromarray(np.asarray((decoded.permute(1,2,0).detach().cpu())*255,dtype=np.uint8))
        OutPutBasePath='/home/m813r/Projects/NeuralImageCompression/BitRateEval/KaggleTestData/SQLC'
        os.makedirs(OutPutBasePath,exist_ok=True)
        StainConv='IndividualRun'
        CompressionModel='Original'
        OutPath=os.path.join(OutPutBasePath,str(lm))
        os.makedirs(OutPath,exist_ok=True)

        transformations=transforms.Compose([transforms.Resize(224),transforms.Lambda(StainDeconvHistomics),transforms.ToTensor(),transforms.Lambda(SQLCCompression)])
        path = '/home/m813r/Projects/NeuralImageCompression/SegmentationBaseline/nnUNET/0/imgs4compression_bad_imgs_excluded' # path to compress segmentation files
        files=glob.glob('/home/m813r/DataSSD/KaggleDataSets/BreaKHis/archive/BreaKHis_400X/test_cropped/*/*.png')### but for perceptual files i need the glob glob operation
        #_, _, files = next(os.walk(path))
        for file in tqdm(files):
            #png_file = Image.open(os.path.join(path, file))
            png_file = Image.open(os.path.join(file))# this was added
            image=transformations(png_file)
            file=os.path.basename(file)##added to get the file name (reason was that due to split in to benign and malignant folders i could not get the file name)
            save_name=os.path.join(OutPath,file)
            x, y = (224, 224)  # 256    map(int, coords.split('_'))
            outtransform = transforms.Resize((y, x))
            image = outtransform(image)
            image.save(save_name)

#CompressSQLC()
#DecodeSQLC()
CompressCAImodels()





