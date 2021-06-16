import torch
import os
from enum import Enum
from typing import Union

from dataclasses import dataclass,asdict

ROOT_WORKSPACE: str=""

class ModelsAvailable(Enum):
    resnet50="resnet50"
    densenet121="densenet121"
    vgg16="vgg16"
    
class Dataset (Enum):
    cifar_crop="cifar-10-diff6cropped.csv"
    cifar_replace="cifar-10-diff6replace.csv"
    cifar_ref="cifar-10.Diff6.RefClass.csv"
    fashionmnist_noref="Fashion-MNIST.Diff6.NoRefClass.csv"
    fashionmnist_ref="Fashion-MNIST.Diff6.RefClass.csv"
    mnist784_ref="mnist_784V2.Diff6.RefClass.csv"
    umistfaces_ref="UMIST_Faces_Cropped.Diff6.RefClass.csv"
    mnist784_classifier="mnist_784V2.Clasification.csv"
    
    
class Optim(Enum):
    adam=1
    sgd=2
    

@dataclass
class CONFIG(object):
    
    experiment=ModelsAvailable.resnet50
    experiment_name:str=experiment.name
    experiment_net:str=experiment.value
    PRETRAINED_MODEL:bool=True
    only_train_head:bool=False #solo se entrena el head

    #torch config
    batch_size:int = 1024
    dataset=Dataset.mnist784_classifier
    dataset_name:str=dataset.name
    precision_compute:int=32
    optim=Optim.adam
    optim_name:str=optim.name
    lr:float = 0.01
    AUTO_LR :bool= False
    # LAMBDA_IDENTITY = 0.0
    NUM_WORKERS:int = 0
    SEED:int=1
    IMG_SIZE:int=28
    NUM_EPOCHS :int= 50
    LOAD_MODEL :bool= True
    SAVE_MODEL :bool= True
    PATH_CHECKPOINT: str= os.path.join(ROOT_WORKSPACE,"/model/checkpoint")
    
    ##model
    features_out_layer1:int=1
    features_out_layer2:int=64
    features_out_layer3:int=256
    tanh1:bool=False
    tanh2:bool=False
    dropout1:float=0.2
    dropout2:float=0.3
    is_mlp_preconfig:bool=False
    
    ##data
    path_data:str=r"/home/dcast/adversarial_project/openml/data"
    
    gpu0:bool=False
    gpu1:bool=True
    notes:str=" correr diferentes modelos para probar su funcionamiento"

def create_config_dict(instance:CONFIG):
    return asdict(instance)