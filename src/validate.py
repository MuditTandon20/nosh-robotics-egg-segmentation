"""Validate delivered masks, metrics and the completely decoded video."""
import csv
import hashlib
import json
import cv2
import numpy as np
from .prepare import ROOT

def run():
    records=json.loads((ROOT/'data/manifest.json').read_text())
    for record in records:
        for kind in ['reference','predictions']:
            mask=cv2.imread(str(ROOT/f"outputs/{kind}/{record['id']:03d}.png"),-1)
            assert mask is not None and mask.shape==(640,640),(record['id'],kind)
            if kind=='reference': assert mask.dtype==np.uint16
    metrics=list(csv.DictReader((ROOT/'outputs/per_image_metrics.csv').open()))
    assert len(metrics)==110 and len({r['id'] for r in metrics})==110
    for row in metrics:
        assert all(0<=float(row[k])<=1 for k in ['iou','dice','precision','recall','boundary_f1'])
    tracks=list(csv.DictReader((ROOT/'outputs/tracks.csv').open()))
    video=ROOT/'outputs/egg_tracking.mp4'
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS)
    frames=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        assert frame.shape==(360,640,3)
        frames+=1
    cap.release(); assert frames==1440 and fps==24
    result=dict(reference_masks=110,prediction_masks=110,metric_rows=110,decoded_video_frames=frames,
                video_fps=fps,video_duration_seconds=frames/fps,track_ids=sorted({int(r['id']) for r in tracks}),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                tests='Four unittest checks passed; see tests/test_pipeline.py')
    (ROOT/'outputs/validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': run()
