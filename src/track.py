"""Classical segmentation + Kalman prediction + gated Hungarian assignment."""
import argparse
import csv
import json
from dataclasses import replace
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .classical import Config, segment
from .prepare import ROOT

class Track:
    def __init__(self, identity, center, frame):
        self.id=identity; self.last_seen=frame; self.hits=1
        self.filter=cv2.KalmanFilter(4,2)
        self.filter.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.filter.measurementMatrix=np.eye(2,4,dtype=np.float32)
        self.filter.processNoiseCov=np.diag([1,1,3,3]).astype(np.float32)
        self.filter.measurementNoiseCov=np.eye(2,dtype=np.float32)*5
        self.filter.errorCovPost=np.eye(4,dtype=np.float32)*25
        self.filter.statePost=np.array([*center,0,0],np.float32).reshape(4,1)
        self.center=np.array(center,dtype=float)
        self.history=[tuple(center)]

    def predict(self):
        self.center=self.filter.predict()[:2,0].astype(float)
        # A hand can hide a stopped egg; damp extrapolation instead of drifting away.
        self.filter.statePost[2:]*=.90

    def update(self,center,frame):
        self.center=self.filter.correct(np.array(center,np.float32).reshape(2,1))[:2,0]
        self.last_seen=frame; self.hits+=1
        self.history=(self.history+[tuple(self.center)])[-36:]

class Tracker:
    def __init__(self, gate=100, max_missed=72):
        self.tracks=[]; self.next_id=1; self.gate=gate; self.max_missed=max_missed

    def update(self,centers,frame):
        self.tracks=[t for t in self.tracks if frame-t.last_seen<=self.max_missed]
        for track in self.tracks:
            if frame-track.last_seen>3:
                track.filter.statePost[2:]=0
            track.predict()
        matches={}; used=set()
        if self.tracks and len(centers):
            cost=np.array([[np.linalg.norm(t.center-c) for c in centers] for t in self.tracks])
            gated=np.where(cost<=self.gate,cost,1e6)
            rows,cols=linear_sum_assignment(gated)
            for r,c in zip(rows,cols):
                if gated[r,c]>=1e6: continue
                self.tracks[r].update(centers[c],frame); matches[c]=self.tracks[r]; used.add(c)
        for index,center in enumerate(centers):
            if index in used: continue
            track=Track(self.next_id,center,frame); self.next_id+=1
            self.tracks.append(track); matches[index]=track
        return matches

def run(input_path,output_path):
    cv2.setNumThreads(1)
    selected=json.loads((ROOT/'outputs/selected_config.json').read_text())
    # Video-specific domain adaptation, documented separately from still-image evaluation.
    # Restrict to tabletop, exclude shirt/logo; stricter white-shell color and smaller objects.
    config=replace(Config(**selected['parameters']),method='hsv',saturation=75,value=170,
                   min_area=.006,max_area=.035,circularity=.55,solidity=.82,close=3,grabcut=0)
    cap=cv2.VideoCapture(str(input_path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {input_path}')
    fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(3)); h=int(cap.get(4))
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened(): raise RuntimeError('MP4 writer failed')
    tracker=Tracker(gate=.40*w,max_missed=192)
    frame_index=0; rows=[]; histogram={}; annotated=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        mask=segment(frame,config).astype(np.uint8)
        mask[:int(.42*h)]=0; mask[int(.90*h):]=0
        mask[:,:int(.02*w)]=0; mask[:,int(.94*w):]=0
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=[c for c in contours if .006*w*h<cv2.contourArea(c)<.035*w*h]
        centers=[]
        for contour in contours:
            m=cv2.moments(contour); centers.append(np.array([m['m10']/m['m00'],m['m01']/m['m00']]))
        matches=tracker.update(centers,frame_index)
        overlay=frame.copy(); overlay[mask>0]=(80,210,80)
        view=cv2.addWeighted(overlay,.35,frame,.65,0)
        for index,contour in enumerate(contours):
            track=matches[index]; x,y,bw,bh=cv2.boundingRect(contour)
            color=tuple(int(v) for v in np.random.default_rng(track.id).integers(80,255,3))
            cv2.rectangle(view,(x,y),(x+bw,y+bh),color,2)
            cv2.putText(view,f'Egg {track.id}',(x,max(15,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            if len(track.history)>1: cv2.polylines(view,[np.array(track.history,np.int32)],False,color,1)
            rows.append(dict(frame=frame_index,time=frame_index/fps,id=track.id,x=float(centers[index][0]),
                             y=float(centers[index][1]),area=cv2.contourArea(contour),confirmed=track.hits>=3))
        cv2.putText(view,f'{frame_index/fps:.1f}s | visible eggs: {len(centers)}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,240,240),2)
        writer.write(view)
        histogram[len(centers)]=histogram.get(len(centers),0)+1
        if frame_index in [288,480,576,672,768,864,960]: annotated.append((frame_index,view.copy()))
        frame_index+=1
    cap.release(); writer.release()
    with (ROOT/'outputs/tracks.csv').open('w',newline='') as stream:
        fields=['frame','time','id','x','y','area','confirmed']
        csvwriter=csv.DictWriter(stream,fieldnames=fields); csvwriter.writeheader(); csvwriter.writerows(rows)
    summary=dict(frames=frame_index,fps=fps,width=w,height=h,total_ids=tracker.next_id-1,
                 detection_count_histogram=histogram,parameters=configuration_dict(config),
                 gate=.40*w,max_missed=192,source='https://www.youtube.com/watch?v=Y4JyV94y2_A',
                 audio='Output is silent; every source frame is retained.',
                 limitation='IDs are not claimed valid across long full occlusions, exits or cuts. No MOT ground truth is available.')
    (ROOT/'outputs/tracking_summary.json').write_text(json.dumps(summary,indent=2))
    for index,frame in annotated: cv2.imwrite(str(ROOT/f'outputs/tracking_{index:04d}.jpg'),frame)
    print(json.dumps(summary,indent=2))

from pathlib import Path
from .classical import configuration_dict
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input',default=str(ROOT/'data/source_video.mp4'))
    parser.add_argument('--output',default=str(ROOT/'outputs/egg_tracking.mp4'))
    args=parser.parse_args(); run(args.input,args.output)
