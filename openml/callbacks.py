import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_grad_cam
import pytorch_lightning as pl
import seaborn as sns
import torch
from PIL import Image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_lightning.callbacks.base import Callback
from seaborn.palettes import color_palette
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split

import wandb


class SplitDatasetWithKFoldStrategy(Callback):
    
    def __init__(self,folds,repetitions,dm,only_train_and_test=False) -> None:
        super().__init__()
        self.folds=folds
        self.repetitions=repetitions
        self.train_val_dataset_initial=torch.utils.data.ConcatDataset([dm.data_train,dm.data_val])
        self.only_train_and_test=only_train_and_test
        kf = KFold(n_splits=folds)

        self.indices_folds={}
        
        for fold, (train_ids, test_ids) in enumerate(kf.split(self.train_val_dataset_initial)):
            self.indices_folds[fold]={
                "train_ids":train_ids,
                "test_ids":test_ids
            }
        self.current_fold=0   

    def create_fold_dataset(self,num_fold,trainer,pl_module):
        
        train_ids=self.indices_folds[num_fold]["train_ids"]
        test_ids=self.indices_folds[num_fold]["test_ids"]
        trainer.datamodule.data_train=torch.utils.data.Subset(self.train_val_dataset_initial,train_ids)
        trainer.datamodule.data_val=torch.utils.data.Subset(self.train_val_dataset_initial,test_ids)
    
    def create_all_train_dataset(self,trainer):
        
        trainer.datamodule.data_val=self.train_val_dataset_initial
        trainer.datamodule.data_train=self.train_val_dataset_initial

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.only_train_and_test:
            self.create_all_train_dataset(trainer)
        else:
            self.create_fold_dataset(pl_module.num_fold,trainer,pl_module)
        return super().on_train_start(trainer, pl_module)

class gradCAMRegressorOneChannel(pytorch_grad_cam.GradCAM):
    def __init__(self, model,
                 target_layer, 
                 use_cuda, 
                 reshape_transform=None):
        super().__init__(model, target_layer, use_cuda=use_cuda, reshape_transform=reshape_transform)

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i]
        return loss
    
class PredictionPlotsAfterTrain(Callback):
    
    def __init__(self,split:str=None) -> None:
        super(PredictionPlotsAfterTrain,self).__init__()
        self.all_test_pred=pd.DataFrame()
        self.split=split
        self.folder_images="/home/dcast/adversarial_project/openml/results"
        self.prefix=split
        
    def _generate_df_from_split(self,trainer: 'pl.Trainer',pl_module:'pl.LightningModule'):
        for batch in self.dataloader:
            if len(batch)==3:
                image,target,idx=batch
            elif len(batch)==4:
                image,target,idx,labels=batch
            with torch.no_grad():
                results=pl_module(image.to(device=pl_module.device))
            if len(batch)==3:
                valid_pred_df = pd.DataFrame({
                    # "image":image.cpu().numpy()[:,0],
                    "target":target.cpu().numpy()[:,0],
                    "results":results.cpu().numpy()[:,0],
                    "id_image": idx,
                    # "norm_PRED_Dffclt": valid_pred[:, 0],
                    # "norm_PRED_Dscrmn": valid_pred[:, 1],
                })
            elif len(batch)==4:
               
                valid_pred_df = pd.DataFrame({
                    # "image":image.cpu().numpy()[:,0],
                    "target":target.cpu().numpy()[:,0],
                    "results":results.cpu().numpy()[:,0],
                    "labels":labels.cpu().numpy()[:,0],
                    "id_image": idx,
                    # "norm_PRED_Dffclt": valid_pred[:, 0],
                    # "norm_PRED_Dscrmn": valid_pred[:, 1],
                })
            self.all_test_pred=pd.concat([self.all_test_pred,valid_pred_df])
        corr=self.all_test_pred.corr(method="spearman")        
        mae=mean_absolute_error(self.all_test_pred["target"],self.all_test_pred["results"])
        mae_relative=mae/self.all_test_pred["target"].std()
        mse=mean_squared_error(self.all_test_pred["target"],self.all_test_pred["results"])
        mse_relative=mse/self.all_test_pred["target"].std()
        trainer.logger.experiment.log({
            "CorrSpearman "+self.prefix:corr.iloc[0,1],
            "mae "+self.prefix:mae,
            "mae relative "+self.prefix: mae_relative,
            "mse "+self.prefix :mse ,
            "mse relative "+self.prefix :mse_relative ,
            
                })
        self.all_test_pred["rank_target"]=self.all_test_pred.target.rank(method="average")
        self.all_test_pred["rank_results"]=self.all_test_pred.results.rank(method="average")
        self.all_test_pred=self.all_test_pred.sort_values("rank_target").reset_index(drop=True)

        ##plotear imágenes dificiles
        df_sorted_hard=self.all_test_pred.sort_values("target",ascending=False).head(5)
        text=self.prefix+" higher"
        self.generate_images_and_upload(trainer,df_sorted_hard,text=text)
        ##plotear imágenes fáciles
        df_sorted_easy=self.all_test_pred.sort_values("target",ascending=True).head(5)
        text=self.prefix+" lowest"
        self.generate_images_and_upload(trainer,df_sorted_easy,text=text)
        
        self.all_test_pred["error"]=(self.all_test_pred["target"]-self.all_test_pred["results"]).abs()
      
        df_sorted_less_error=self.all_test_pred.sort_values("error",ascending=True).head(5)
        # print(df_sorted_less_error)
        # self.__generate_image_with_grad_cam(df_sorted_hard,trainer,pl_module,"hard")
        # self.__generate_image_with_grad_cam(df_sorted_easy,trainer,pl_module,"easy")
        # self.__generate_image_with_grad_cam(df_sorted_less_error,trainer,pl_module,"minor_error")
        
        # self._plots_scatter_rank_plot(trainer) #reactivate
        
        
        ##plotear imágenes dificiles que han sido predichas como fáciles
        self.all_test_pred["difference"]=self.all_test_pred["target"]-self.all_test_pred["results"]
        df_images_predict_easy_but_the_true_is_there_are_hard= self.all_test_pred.sort_values("difference",ascending=False).head(5)
        text=self.prefix+" Hard but predict easy"
        self.generate_images_and_upload(trainer,df_images_predict_easy_but_the_true_is_there_are_hard,text=text)
        ##plotear imágenes fáciles que han sido predichas como dificiles

        df_images_predict_hard_but_the_true_is_there_are_easy= self.all_test_pred.sort_values("difference",ascending=True).head(5)
        # print(df_images_predict_hard_but_the_true_is_there_are_easy)
        text=self.prefix+" Easy but predict hard"
        self.generate_images_and_upload(trainer,df_images_predict_hard_but_the_true_is_there_are_easy,text=text)
        
    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.split=="train":
            self.dataloader=trainer.datamodule.train_dataloader()
        elif self.split=="val":
            self.dataloader=trainer.datamodule.val_dataloader()
        elif self.split=="test":
            self.dataloader=trainer.datamodule.test_dataloader()
        

        self._generate_df_from_split(trainer,pl_module)
   
        return super().on_train_end(trainer, pl_module)
    
    def __generate_image_with_grad_cam(self,df,trainer:'pl.Trainer',pl_module,text):
        def revert_normalization(img,mean,std):
            return (img*std+mean)
        target_layer=list(pl_module.model.children())[-4] #con menos 4 funciona
        cam=gradCAMRegressorOneChannel(model=pl_module,target_layer=target_layer,use_cuda=True)
        df=df.head(5)
        normalize=trainer.datamodule.data_train.dataset.transform.transforms[0]
        # normalize=
        mean=normalize.mean
        std=normalize.std
        if "labels" in df.columns:
            iterator=zip(df.id_image,df.labels,df.target,df.results)
        else:
            iterator=zip (df.id_image, df.target,df.results)
        for batch in iterator:  
            if len(batch)==3:#isinstance(batch,int):
                idx,target,results=batch
                label=None
                
            else:
                idx,label,target,results=batch     
            image=torch.unsqueeze(self.dataloader.dataset.dataset._create_image_from_dataframe(idx),dim=0).to(device=pl_module.device)
            grayscale_cam=cam(input_tensor=image,eigen_smooth=False)#si no funciona poner en True
            
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]

            if np.isnan(grayscale_cam).any():
                print("hola, alguno es nan")

            image=image.cpu().numpy()
            gray=image[0,:,:,:]
            gray=np.moveaxis(gray,0,-1)
            if gray.shape[-1]!=3:
               
                img_bw_with_3_channels=cv2.merge((gray,gray,gray))
                img_to_save=np.uint8((img_bw_with_3_channels+1)*127.5)
            else:
                img_bw_with_3_channels=gray    
                img_bw_with_3_channels=revert_normalization(img_bw_with_3_channels,mean,std)
                img_to_save=np.uint8(img_bw_with_3_channels*255)
            # 
            img=Image.fromarray(img_to_save)
            img.save(os.path.join(self.folder_images,f"{text} {idx} image_3_channel.png"))
            heatmap=cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_PLASMA   )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap)
            cv2.imwrite(os.path.join(self.folder_images,f'{text} {idx} heatmap_cam.jpg'), heatmap)
            alpha=0.5
            beta=(1.0-alpha)
            
            dst = np.uint8(alpha*(heatmap)+beta*(img_to_save))
            cv2.imwrite(os.path.join(self.folder_images,f"{text} {idx} mixture.jpg"), dst)
            visualization = pytorch_grad_cam.utils.image.show_cam_on_image(img_bw_with_3_channels,
                                                                        grayscale_cam,use_rgb=True,
                                                                        colormap=cv2.COLORMAP_PLASMA  )
            img=Image.fromarray(visualization)
            
            img.save(os.path.join(self.folder_images,f"{text} {idx} probando.png"))
            trainer.logger.experiment.log({
                "graficas gradcam "+self.prefix:wandb.Image(img,caption=f" {idx} grad cam, Label {label}, Target: {target}, Pred: {results} "),
                            })
            
    def _plots_scatter_rank_plot(self,trainer:'pl.Trainer'):
        self._bar_rank_plot(trainer,
                            xlabel1="valores ordenador por target",
                            xlabel2="valores ordenador por results",
                            ylabel="puesto en el ranking",
                            title="grafico de barras para correlacionar valores por ranking")
        
        if "labels" in self.all_test_pred.columns:
            self.all_test_pred.to_csv("/home/dcast/adversarial_project/openml/results_to_Carlos.csv")
            self.all_test_pred=self.all_test_pred.sample(frac=0.01)
            self._scatter_plot(x=self.all_test_pred.target,
                           y=self.all_test_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion",
                           labels=self.all_test_pred.labels)
        else:
            self._scatter_plot(x=self.all_test_pred.target,
                           y=self.all_test_pred.results,
                           xname="target",
                           yname="results",
                           trainer=trainer,
                           title="Grafico de dispersion") 
        

    def _scatter_plot(self,x,y,xname,yname,trainer,title,labels=None):
        alpha=None
        fig = plt.figure(figsize=(14,7))
        if labels is None:
            # plt.scatter(x=x,y=y,alpha=alpha)
            sns.scatterplot(x=x,y=y, alpha=alpha)
        else:
            # plt.scatter(x=x,y=y,c=labels,alpha=alpha)
            color_pallete=sns.color_palette("tab10",n_colors=10) #un error extraño
            sns.scatterplot(x=x,y=y,hue=labels,alpha=alpha,palette=color_pallete)
        plt.title(title)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.xlim([-6,6])
        plt.ylim([-6,6])
        plt.savefig("algo.jpg")
        trainer.logger.experiment.log({
            "graficas scatter "+self.prefix:wandb.Image(fig,caption="scatter plot"),
        })
        plt.close()
        
    def _bar_rank_plot(self,trainer,xlabel1,xlabel2,ylabel,title):
        fig = plt.figure(figsize=(14,7))
        plt.bar(self.all_test_pred.index,height=self.all_test_pred.rank_target)
        plt.bar(self.all_test_pred.index,height=self.all_test_pred.rank_results)
        plt.title(title)
        plt.xlabel("valores ordenados por Dffclt")
        plt.xlabel("valores ordenados por confidence")
        plt.ylabel("puesto en el ranking")
        trainer.logger.experiment.log({
            "graficas rank "+self.prefix:wandb.Image(fig,caption="rank plot"),
            # "global_step": trainer.global_step
        })
        plt.close()

    def generate_images_and_upload(self,trainer,df:pd.DataFrame,text:str):
        pass
    
        # images=[]
        # for idx in df.id_image:
        #     images.append(self.dataloader.dataset.dataset._create_image_from_dataframe(idx))
        # if "labels" in df.columns:
        #     trainer.logger.experiment.log({
        #         f"{text}/examples": [
        #             wandb.Image(x, caption=f"Pred:{round(pred,4)}, Label:{round(target,4)}, Num: {label}") 
        #                 for x, pred, target,label in zip(images, df.results, df.target,df.labels)
        #             ],
        #         })
        # else:
        #     trainer.logger.experiment.log({
        #         f"{text}/examples": [
        #             wandb.Image(x, caption=f"Pred:{round(pred,4)}, Label:{round(target,4)}") 
        #                 for x, pred, target in zip(images, df.results, df.target)
        #             ],
        #         })
