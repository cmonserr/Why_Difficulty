
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from timm.data.constants import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
from torchvision.transforms.transforms import ToTensor
from albumentations.pytorch import ToTensorV2

class Loader(Dataset):
    
    def __init__(self,dir_csv_file:str,reshape_shape:tuple,einum_reshape:str,transform:A.Compose) -> None:
        super().__init__()
        
        self.dir_csv_file=dir_csv_file
        self.data=pd.read_csv(self.dir_csv_file,index_col="Unnamed: 0")

        if "X" in self.data.columns:
            self.data.drop(columns="X",inplace=True) #x son las filas que Nando no ha eliminado, el indice original
        if "class" in self.data.columns:
            self.labels=self.data.pop("class").to_numpy()
        else:
            self.labels=None
        if "Hard" in self.data.columns:
            self.y=self.data.pop("Hard").to_numpy()
            self.data.pop("Dffclt")
        else:
            self.y=self.data.pop("Dffclt").to_numpy()
        self.data=self.data
        self.reshape_shape=reshape_shape
        self.einum_reshape=einum_reshape
        self.transform=transform
        
    def __getitem__(self, index):
        img=self._create_image_from_dataframe(index)
        
        target=torch.tensor(self.y[index],dtype=torch.float)
        target=torch.unsqueeze(target,0)
        #pendiente aplicar transform simple a example
        if self.labels is None:
            return img,target,index
        else:
            label=torch.tensor(self.labels[index],dtype=int)
            label=torch.unsqueeze(label,0)
            return img,target,index,label
    def _create_image_from_dataframe(self,index):
        example=self.data.iloc[index]
        example=np.array(example,dtype=int)
        example=example.reshape(self.reshape_shape)  
        example = np.einsum(self.einum_reshape, example)
        augmentations=self.transform(image=example)
        img=augmentations["image"]
        return img
    def __len__(self):
        
        return self.data.shape[0]
    
    
class Cifar10Loader(Loader):
    def __init__(self, dir_csv_file: str) -> None:
        reshape_shape=(3,32,32)
        einum_reshape='ijk->jki'
        transform=A.Compose(
        [
            A.Normalize(
                # mean=[0, 0, 0],
                mean=[IMAGENET_DEFAULT_MEAN[0], IMAGENET_DEFAULT_MEAN[1], IMAGENET_DEFAULT_MEAN[2]],
                # std=[1, 1, 1],
                std=[IMAGENET_DEFAULT_STD[0], IMAGENET_DEFAULT_STD[1], IMAGENET_DEFAULT_STD[2]],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                
                )
        super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        
class FashionMnistLoader(Loader):
    def __init__(self, dir_csv_file: str) -> None:
        reshape_shape=(28,28)
        einum_reshape='ij->ij'
        transform=A.Compose(
        [
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                )
        super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        
class MnistLoader(Loader):
    def __init__(self, dir_csv_file: str) -> None:
        reshape_shape=(28,28) #revisar
        einum_reshape='ij->ij'
        transform=A.Compose(
        [
            A.Normalize(
                mean=[0.5],
                std=[0.5],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                )
        super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        
class UMISTFacesLoader(Loader):  
    def __init__(self, dir_csv_file: str) -> None:
        reshape_shape=(112,92) #revisar
        einum_reshape='ij->ij'
        transform=A.Compose(
        [
            A.Normalize(
                # mean=[0, 0, 0],
                mean=[0.5],
                # std=[1, 1, 1],
                std=[0.5],
                max_pixel_value=255,
                ),
            ToTensorV2(),
                ]
                
                )
        super().__init__(dir_csv_file,reshape_shape,einum_reshape,transform)
        

        