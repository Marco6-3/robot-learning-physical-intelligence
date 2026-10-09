"""Factorial causal delayed-query benchmark; algorithmic, not a physical simulator."""
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from common import *
from models import make_model, count_parameters

def generate(n,seed,prevalence,lag,length=128):
    streams=np.random.SeedSequence(seed).spawn(3)
    key=np.random.default_rng(streams[0]).choice([-1.,1.],size=(n,length)).astype(np.float32)
    query=(np.random.default_rng(streams[1]).random((n,length))<2*prevalence).astype(np.float32)
    distract=np.random.default_rng(streams[2]).normal(size=(n,length,6)).astype(np.float32)
    x=np.concatenate([key[:,:,None],query[:,:,None],distract],axis=-1)
    y=np.zeros((n,length),np.float32); y[:,lag:]=query[:,lag:]*(key[:,:-lag]>0)
    mask=np.zeros(length,np.float32); mask[64:]=1
    return x,y,mask

def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    models=['mlp','tcn','gru','mamba']; seeds=list(range(5))
    if args.smoke: models=['gru','mamba']; seeds=[0]
    manifest={'generator':'causal-delayed-query-v1','length':128,'score_start':64,
              'prevalence':[.01,.20],'lags':[2,64],'nominal_hz':20,
              'train_records':[64,256],'validation_records':256,'test_records':512,
              'train_seeds':seeds,'models':models,'steps':args.steps,
              'learning_rates':[.001,.003],'tune_seed':901,'test_used_for_selection':False,
              'interpretation':'Algorithmic delayed-key retrieval; not physical contact simulation or future forecasting.'}
    save_json(out/'manifest.json',manifest)
    # Hyperparameters selected per factorial cell/model using independent tuning training seed.
    for rate in [.01,.20]:
      for lag in [2,64]:
       for n in [64,256]:
        vx,vy,mask=generate(256,200001,rate,lag)
        ex,ey,_=generate(512,300001,rate,lag)
        vm=np.broadcast_to(mask,vy.shape).astype(bool); em=np.broadcast_to(mask,ey.shape).astype(bool)
        for name in models:
          key=f'p{rate:g}_lag{lag}_n{n}_{name}'
          hp_path=out/(key+'_tuning.json')
          if hp_path.exists(): hp=json.loads(hp_path.read_text())
          else:
            xx,yy,_=generate(n,100901,rate,lag); trials=[]
            for lr in [.001,.003]:
              seed_all(901); model=make_model(name,8)
              model,meta=train(model,xx,yy,mask,vx,vy,mask,901,lr,steps=args.steps)
              trials.append({'lr':lr,**meta})
              del model
            hp={'selected_lr':max(trials,key=lambda z:z['best_val_ap'])['lr'],'trials':trials}
            save_json(hp_path,hp)
            print(json.dumps({'tuned':key,'lr':hp['selected_lr'],'val':[v['best_val_ap'] for v in trials]}),flush=True)
          for seed in seeds:
            path=out/(key+f'_seed{seed}.json')
            if path.exists(): continue
            xx,yy,_=generate(n,100000+seed,rate,lag)
            seed_all(seed); model=make_model(name,8)
            model,meta=train(model,xx,yy,mask,vx,vy,mask,seed,hp['selected_lr'],steps=args.steps)
            vp=predict(model,vx); threshold=best_threshold(vy[vm],vp[vm])
            ep=predict(model,ex); res=metrics(ey[em],ep[em],threshold)
            qm=em & (ex[:,:,1]>0)
            res['query_ap']=float(average_precision_score(ey[qm],ep[qm])); res['query_n']=int(qm.sum())
            # Mechanism diagnostic: permute key across records, preserving event queries.
            ab=ex.copy(); ab[:,:,0]=np.roll(ab[:,:,0],1,axis=0)
            apred=predict(model,ab)
            res['key_shuffled_ap']=float(average_precision_score(ey[em],apred[em]))
            res['key_shuffled_query_ap']=float(average_precision_score(ey[qm],apred[qm]))
            result={'rate':rate,'lag':lag,'n_train':n,'model':name,'seed':seed,
                    'parameters':count_parameters(model),'metrics':res,'training':meta}
            np.savez_compressed(out/(key+f'_seed{seed}_predictions.npz'),y=ey[:,64:],p=ep[:,64:],query=ex[:,64:,1],key_shuffled_p=apred[:,64:])
            torch.save(model.state_dict(),out/(key+f'_seed{seed}.pt'))
            save_json(path,result)
            print(json.dumps({'completed':key,'seed':seed,'ap':res['ap'],'query_ap':res['query_ap'],'sec':meta['seconds']}),flush=True)
            del model
          if args.smoke: return

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='work/results/synthetic');p.add_argument('--steps',type=int,default=400);p.add_argument('--smoke',action='store_true');run(p.parse_args())
