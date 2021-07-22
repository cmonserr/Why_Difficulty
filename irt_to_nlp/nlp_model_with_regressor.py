
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          GPTNeoForCausalLM)


class CustomGPTNeo(nn.Module):
    
    def __init__(self):
        super(CustomGPTNeo, self).__init__()
        in_dim=50257
        hidden_dim1=450
        num_tokens=1790
        hidden_dim2=1
        out_dim=1
        

        self.gptneo = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

        self.regressor=MLP(in_dim,hidden_dim1,hidden_dim2,out_dim,num_tokens)

        
    def forward(self,ids):
        x=self.gptneo(ids)
        x=x.logits
        x=self.regressor(x)
        
        return x
    
class CustomBertBaseCased(nn.Module):
    
    def __init__(self):
        super(CustomBertBaseCased, self).__init__()
        in_dim=256
        hidden_dim1=32
        # num_tokens=1790
        hidden_dim2=10
        out_dim=1


        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=in_dim)

        self.regressor=MLP(in_dim,hidden_dim1,hidden_dim2,out_dim)

        
    def forward(self,ids):
        x=self.model(ids)
        x=x.logits
        x=self.regressor(x)
        
        return x
    
class CustomDistiledBertBaseCased(nn.Module):
    
    def __init__(self):
        super(CustomDistiledBertBaseCased, self).__init__()
        in_dim=256
        hidden_dim1=16
        # num_tokens=1790
        hidden_dim2=8
        out_dim=1


        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=in_dim)
        self.regressor=MLP(in_dim,hidden_dim1,hidden_dim2,out_dim)

        
    def forward(self,ids):
        x=self.model(ids)
        x=x.logits
        x=self.regressor(x)
        
        return x
    
class CustomDistiledBertBaseCased(nn.Module):
    
    def __init__(self):
        super(CustomDistiledBertBaseCased, self).__init__()
        in_dim=256
        hidden_dim1=32
        # num_tokens=1790
        hidden_dim2=10
        out_dim=1
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=in_dim)
        self.regressor=MLP(in_dim,hidden_dim1,hidden_dim2,out_dim)
        
    def forward(self,ids):
        x=self.model(ids)
        x=x.logits
        x=self.regressor(x)
        
        return x
class CustomDistiledGPT2(nn.Module):
    
    def __init__(self):
        super(CustomDistiledGPT2, self).__init__()
        in_dim=50257
        hidden_dim1=16
        num_tokens=1024
        hidden_dim2=8
        out_dim=1
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2",)
        self.regressor=MLP(in_dim,hidden_dim1,hidden_dim2,out_dim,num_tokens)
        
    def forward(self,ids):
        x=self.model(ids)
        x=x.logits
        x=self.regressor(x)
        
        return x
       
class MLP(nn.Module):
    
    def __init__(self,in_dim,hidden_dim1,hidden_dim2,out_dim,num_tokens=1) -> None:
        super(MLP,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim1),
            nn.Flatten(),
            nn.BatchNorm1d(hidden_dim1*num_tokens),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim1*num_tokens, hidden_dim2),
            nn.ReLU(inplace=True),
            # nn.Flatten(),
            nn.Linear(hidden_dim2, out_dim)
        )
        
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        
        return x
