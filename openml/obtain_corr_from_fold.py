from openml.config import ModelsAvailable,Dataset

import pandas as pd
import os
import glob

def get_list_with_all_csv_with_format(path,lr:str,model:str,dataset:str) ->list:
    #the format is regressor_{repetition}_{fold}_{lr}_{train/val}_{dataset}_{model}.csv
    
    repetition=0
    fold_max=5
    split="val"
    format=f'{path}/regressor*{lr}*{split}*{dataset}_{model}.csv'
    files=glob.glob(format)
    return files

def generate_dataframe_concated(list_csv:list)->pd.DataFrame:
    
    df=pd.DataFrame()
    li=[]
    for csv in list_csv:
        df_aux=pd.read_csv(csv,index_col="Unnamed: 0")
        li.append(df_aux)
        
    df=pd.concat(li, axis=0, ignore_index=True)
    # print(df.head())
    print(df.shape)
    print(df.nunique())
    return df

def generate_corr_and_rank(df:pd.DataFrame):
    
    corr=df.corr(method="spearman")  
    print(corr)
    
path_with_result:str="./openml/data/results"

lr_used:str="0.001"
model=ModelsAvailable.densenet121
model_str:str=model.name
dataset=Dataset.umistfaces_ref
dataset_str:str=dataset.name

    
files=get_list_with_all_csv_with_format(path_with_result,lr_used,model_str,dataset_str)

df=generate_dataframe_concated(files)

generate_corr_and_rank(df)


