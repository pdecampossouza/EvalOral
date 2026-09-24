#!/usr/bin/env python3
"""Reproducible training and Grad-CAM generation for the separate CNN audit branch.

This script follows the manuscript architecture (96x96 RGB, 16/32/64 conv
channels, Adam 2e-3, batch 16). The CNN is NOT an input to the primary fuzzy
classifier. Because the historical trained weights were not retained in the
release bundle, this script is a deterministic retraining protocol rather than
byte-identical restoration of the manuscript pilot weights.
"""
from pathlib import Path
import argparse, json, random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score,accuracy_score
from torch import nn
from torch.utils.data import DataLoader
import torch
from evaloral.data import repo_root,load_view121
from evaloral.gradcam import OralImageDataset,SmallOralCNN,gradcam


def setseed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_epoch(model,loader,opt,device):
    model.train(); lossfn=nn.CrossEntropyLoss(); tot=0.0
    for x,y,_ in loader:
        x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=lossfn(logits,y); loss.backward(); opt.step(); tot+=float(loss.detach().item())*len(x)
    return tot/max(len(loader.dataset),1)


def eval_model(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y,_ in loader:
            p=model(x.to(device)).argmax(1).cpu().numpy(); ps.extend(p); ys.extend(y.numpy())
    return f1_score(ys,ps,average='macro',zero_division=0),accuracy_score(ys,ps)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=20260824);ap.add_argument('--max-selection-epochs',type=int,default=12);ap.add_argument('--force-epoch',type=int,default=None);a=ap.parse_args()
    setseed(a.seed);root=repo_root(); _,_,m=load_view121(root);out=root/'results/generated/09_gradcam';out.mkdir(parents=True,exist_ok=True)
    vols=np.array(sorted(m.volunteer_id.unique()));np.random.default_rng(a.seed).shuffle(vols)
    outer=vols[:7];external=vols[7:]; inner_fit=outer[:5];inner_val=outer[5:]
    fit=m[m.volunteer_id.isin(inner_fit)].copy();val=m[m.volunteer_id.isin(inner_val)].copy();ext=m[m.volunteer_id.isin(external)].copy()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def loader(df,shuffle=False): return DataLoader(OralImageDataset(df,root),batch_size=16,shuffle=shuffle,num_workers=0)
    model=SmallOralCNN().to(device);opt=torch.optim.Adam(model.parameters(),lr=2e-3);hist=[];best_epoch=1;best=-1
    for ep in range(1,a.max_selection_epochs+1):
        loss=train_epoch(model,loader(fit,True),opt,device);f1,acc=eval_model(model,loader(val),device);hist.append({'epoch':ep,'train_loss':loss,'val_macro_f1':f1,'val_accuracy':acc})
        if f1>best: best=f1;best_epoch=ep
    selected=a.force_epoch if a.force_epoch is not None else best_epoch
    # Manuscript pilot selected epoch 1. To reproduce that published operating
    # point exactly, invoke --force-epoch 1.
    setseed(a.seed);model=SmallOralCNN().to(device);opt=torch.optim.Adam(model.parameters(),lr=2e-3);outer_df=m[m.volunteer_id.isin(outer)].copy()
    for _ in range(selected): train_epoch(model,loader(outer_df,True),opt,device)
    f1,acc=eval_model(model,loader(ext),device);torch.save(model.state_dict(),out/'cnn_refit_state.pt')
    pd.DataFrame(hist).to_csv(out/'model_selection_history.csv',index=False)
    summary={'seed':a.seed,'outer_volunteers':outer.tolist(),'external_volunteers':external.tolist(),'selected_epoch':selected,'inner_best_epoch':best_epoch,'external_macro_f1':float(f1),'external_accuracy':float(acc),'device':str(device)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    # Make Grad-CAM for all external cases, then select 12 evenly across confidence.
    ref=pd.read_csv(root/'data/reference/final_rule_assignments_view121.csv')[['image_id','final_rule_id']]
    rows=[];cams={};model.eval()
    for _,r in ext.reset_index(drop=True).iterrows():
        ds=OralImageDataset(pd.DataFrame([r]),root);x,y,i=ds[0];xx=x.unsqueeze(0).to(device);cam,logits,target=gradcam(model,xx);prob=torch.softmax(torch.tensor(logits),0).numpy();pred=int(np.argmax(prob));
        rows.append({'image_id':r.image_id,'released_view':r.released_view,'cnn_prediction':['frontal','occlusal'][pred],'confidence':float(prob[pred]),'correct':int(pred==int(y)),'filepath':r.filepath});cams[r.image_id]=cam
    df=pd.DataFrame(rows).merge(ref,on='image_id',how='left').sort_values('confidence')
    if len(df)>12: sel=df.iloc[np.linspace(0,len(df)-1,12).round().astype(int)].drop_duplicates('image_id')
    else: sel=df
    df.to_csv(out/'external_predictions.csv',index=False);sel.to_csv(out/'gradcam_12_case_table.csv',index=False)
    fig,axes=plt.subplots(3,4,figsize=(11,8));axes=np.asarray(axes).ravel()
    for ax,(_,r) in zip(axes,sel.iterrows()):
        im=np.asarray(Image.open(root/r.filepath).convert('RGB').resize((96,96),Image.Resampling.BILINEAR));cam=cams[r.image_id]
        ax.imshow(im);ax.imshow(cam,cmap='jet',alpha=.42,extent=(0,96,96,0));ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"{r.cnn_prediction} | R{int(r.final_rule_id) if pd.notna(r.final_rule_id) else '?'}\nconf={r.confidence:.2f}",fontsize=8)
    for ax in axes[len(sel):]:ax.axis('off')
    fig.suptitle('Rule-linked Grad-CAM cases from the separate CNN audit branch');fig.tight_layout();fig.savefig(out/'Figure_gradcam_12_cases.png',dpi=300,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
