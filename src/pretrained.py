"""Task 1: pretrained MobileSAM with independent human point prompts."""
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor
from .prepare import ROOT

def run(limit=None, only=None):
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    model=sam_model_registry['vit_t'](checkpoint=str(ROOT/'models/mobile_sam.pt'))
    model.eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictor=SamPredictor(model.to(device))
    records=json.loads((ROOT/'data/manifest.json').read_text())
    folder=ROOT/'outputs/reference'; folder.mkdir(parents=True,exist_ok=True)
    audit=[]
    if only:
        previous=ROOT/'outputs/reference_audit.json'
        if previous.exists(): audit=[r for r in json.loads(previous.read_text())['images'] if r['id'] not in only]
    for record in records[:limit]:
        if only and record['id'] not in only: continue
        start=time.perf_counter()
        image=cv2.imread(str(ROOT/record['path']))
        h,w=image.shape[:2]
        instances=np.zeros((h,w),np.uint16)
        scores=[]
        if record['points']:
            with torch.inference_mode():
                predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                for index, point in enumerate(record['points'],1):
                    # Other egg centers are negative to keep each prediction an instance.
                    coords=np.array([point]+[p for p in record['points'] if p!=point])*[w,h]
                    labels=np.array([1]+[0]*(len(coords)-1))
                    box=np.array(record['boxes'][index-1])*[w,h,w,h] if record.get('boxes') else None
                    masks,quality,_=predictor.predict(point_coords=coords,point_labels=labels,box=box,multimask_output=True)
                    selected=int(np.argmax(quality))
                    instances[masks[selected]]=index
                    scores.append(float(quality[selected]))
        cv2.imwrite(str(folder/f"{record['id']:03d}.png"),instances)
        audit.append(dict(id=record['id'],scores=scores,seconds=time.perf_counter()-start,
                          foreground_fraction=float((instances>0).mean())))
        print(record['id'],scores,round(audit[-1]['seconds'],2),flush=True)
    (ROOT/'outputs/reference_audit.json').write_text(json.dumps(dict(model='MobileSAM vit_t',device=device,
        checkpoint_sha256=hashlib.sha256((ROOT/'models/mobile_sam.pt').read_bytes()).hexdigest(),images=audit),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--limit',type=int); parser.add_argument('--only',type=int,nargs='+')
    args=parser.parse_args(); run(args.limit,args.only)
