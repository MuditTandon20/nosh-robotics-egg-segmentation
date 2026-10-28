"""Development-only ablations, frozen selection, then held-out evaluation."""
import argparse
import csv
import json
import time
import cv2
import numpy as np
from scipy import ndimage
from .prepare import ROOT
from .classical import TRIALS, segment, configuration_dict

def metrics(reference,prediction):
    a=np.asarray(reference,dtype=bool); b=np.asarray(prediction,dtype=bool)
    tp=int((a&b).sum()); fp=int((~a&b).sum()); fn=int((a&~b).sum()); tn=int((~a&~b).sum())
    union=tp+fp+fn
    ab=a^ndimage.binary_erosion(a); bb=b^ndimage.binary_erosion(b)
    if not ab.any() and not bb.any(): boundary=1.
    elif not ab.any() or not bb.any(): boundary=0.
    else:
        precision=float((ndimage.distance_transform_edt(~ab)[bb]<=3).mean())
        recall=float((ndimage.distance_transform_edt(~bb)[ab]<=3).mean())
        boundary=2*precision*recall/max(precision+recall,1e-12)
    return dict(iou=tp/union if union else 1.,dice=2*tp/(2*tp+fp+fn) if union else 1.,
                precision=tp/(tp+fp) if tp+fp else float(not a.any()),
                recall=tp/(tp+fn) if tp+fn else float(not b.any()),
                boundary_f1=boundary,tp=tp,fp=fp,fn=fn,tn=tn)

def summarize(rows):
    keys=['iou','dice','precision','recall','boundary_f1','seconds']
    result={key:float(np.mean([r[key] for r in rows])) for key in keys}
    tp=sum(r['tp'] for r in rows); fp=sum(r['fp'] for r in rows); fn=sum(r['fn'] for r in rows)
    result['micro_iou']=tp/max(tp+fp+fn,1)
    groups=sorted(set(r['group'] for r in rows)); rng=np.random.default_rng(2026)
    # Cluster bootstrap: sample entire scene groups to respect within-scene dependence.
    samples=[]
    for _ in range(2000):
        selected=rng.choice(groups,len(groups),replace=True)
        values=[r['iou'] for g in selected for r in rows if r['group']==g]
        samples.append(np.mean(values))
    result['iou_cluster_bootstrap_95ci']=np.percentile(samples,[2.5,97.5]).tolist()
    result['images']=len(rows); result['groups']=len(groups)
    return result

def run(dev_only=False):
    cv2.setNumThreads(1); cv2.setRNGSeed(2026)
    records=json.loads((ROOT/'data/manifest.json').read_text())
    development=[r for r in records if r['split']=='dev']
    evaluation=[r for r in records if r['split']=='eval']
    logs=[]; all_dev={}
    for name,config,reason in TRIALS:
        rows=[]
        for record in development:
            image=cv2.imread(str(ROOT/record['path']))
            reference=cv2.imread(str(ROOT/f"outputs/reference/{record['id']:03d}.png"),-1)>0
            start=time.perf_counter(); prediction=segment(image,config); seconds=time.perf_counter()-start
            rows.append(dict(id=record['id'],group=record['group'],split='dev',seconds=seconds,**metrics(reference,prediction)))
        summary=summarize(rows)
        logs.append(dict(trial=name,reason=reason,parameters=configuration_dict(config),development=summary))
        all_dev[name]=rows
        print(name,round(summary['iou'],4),flush=True)
    winner=max(logs,key=lambda x:x['development']['iou'])['trial']
    for trial in logs:
        trial['decision']='kept' if trial['trial']==winner else 'discarded'
        trial['decision_reason']='Highest development mean IoU' if trial['trial']==winner else 'Lower development mean IoU than selected configuration'
    config=next(c for n,c,_ in TRIALS if n==winner)
    (ROOT/'outputs/selected_config.json').write_text(json.dumps(dict(trial=winner,parameters=configuration_dict(config)),indent=2))
    (ROOT/'outputs/experiments.json').write_text(json.dumps(logs,indent=2))
    if dev_only:
        print('Selected on development:',winner); return
    # Eval labels first enter the classical experiment only after the configuration is saved.
    rows=list(all_dev[winner]); folder=ROOT/'outputs/predictions'; folder.mkdir(exist_ok=True)
    for record in records:
        image=cv2.imread(str(ROOT/record['path']))
        start=time.perf_counter(); prediction=segment(image,config); seconds=time.perf_counter()-start
        cv2.imwrite(str(folder/f"{record['id']:03d}.png"),prediction.astype(np.uint8)*255)
        if record['split']=='eval':
            reference=cv2.imread(str(ROOT/f"outputs/reference/{record['id']:03d}.png"),-1)>0
            rows.append(dict(id=record['id'],group=record['group'],split='eval',seconds=seconds,**metrics(reference,prediction)))
    with (ROOT/'outputs/per_image_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={s:summarize([r for r in rows if r['split']==s]) for s in ['dev','eval']}
    summary['selected_trial']=winner
    summary['eval_by_scene']={g:summarize([r for r in rows if r['group']==g]) for g in sorted(set(r['group'] for r in evaluation))}
    (ROOT/'outputs/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--dev-only',action='store_true')
    run(parser.parse_args().dev_only)
