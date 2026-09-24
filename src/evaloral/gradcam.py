from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

class OralImageDataset(Dataset):
    def __init__(self,meta,root,labels=True):self.meta=meta.reset_index(drop=True);self.root=Path(root);self.labels=labels
    def __len__(self):return len(self.meta)
    def __getitem__(self,i):
        r=self.meta.iloc[i]
        im=Image.open(self.root/r.filepath).convert('RGB').resize((96,96),Image.Resampling.BILINEAR)
        x=torch.from_numpy(np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.0)
        if not self.labels:return x,r.image_id
        y=0 if r.released_view=='frontal' else 1
        return x,torch.tensor(y,dtype=torch.long),r.image_id

class SmallOralCNN(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(3,16,3,padding=1);self.conv2=nn.Conv2d(16,32,3,padding=1);self.conv3=nn.Conv2d(32,64,3,padding=1);self.pool=nn.MaxPool2d(2);self.relu=nn.ReLU();self.gap=nn.AdaptiveAvgPool2d(1);self.fc1=nn.Linear(64,32);self.fc2=nn.Linear(32,2)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)));x=self.pool(self.relu(self.conv2(x)));x=self.relu(self.conv3(x));x=self.gap(x).flatten(1);x=self.relu(self.fc1(x));return self.fc2(x)


def gradcam(model,x,target=None):
    model.eval();acts={};grads={}
    def fh(m,inp,out):acts['v']=out;out.register_hook(lambda g:grads.__setitem__('v',g))
    h=model.conv3.register_forward_hook(fh)
    logits=model(x);target=int(logits.argmax(1)[0]) if target is None else int(target);model.zero_grad(set_to_none=True);logits[0,target].backward();h.remove()
    a=acts['v'][0];g=grads['v'][0];w=g.mean(dim=(1,2),keepdim=True);cam=(w*a).sum(0).clamp(min=0);cam-=cam.min();cam/=cam.max().clamp(min=1e-8);return cam.detach().cpu().numpy(),logits.detach().cpu().numpy()[0],target
