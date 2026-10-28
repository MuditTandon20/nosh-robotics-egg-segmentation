import unittest
import json
import cv2
import numpy as np
from src.prepare import ROOT
from src.evaluate import metrics
from src.classical import Config, segment
from src.track import Tracker

class PipelineTests(unittest.TestCase):
    def test_metrics_known_overlap(self):
        a=np.array([[1,1],[0,0]],bool); b=np.array([[1,0],[1,0]],bool)
        result=metrics(a,b)
        self.assertAlmostEqual(result['iou'],1/3)
        self.assertAlmostEqual(result['dice'],.5)
        self.assertEqual(metrics(a,a)['boundary_f1'],1)
        self.assertEqual(metrics(a*0,a*0)['iou'],1)

    def test_split_integrity(self):
        records=json.loads((ROOT/'data/manifest.json').read_text())
        self.assertEqual(sum(r['split']=='eval' for r in records),22)
        dev={r['group'] for r in records if r['split']=='dev'}
        evaluation={r['group'] for r in records if r['split']=='eval'}
        self.assertFalse(dev&evaluation)
        self.assertEqual(len(set(r['sha256'] for r in records)),110)
        self.assertTrue(all((ROOT/f"outputs/reference/{r['id']:03d}.png").exists() for r in records))

    def test_two_eggs_on_dark_background(self):
        image=np.full((240,320,3),25,np.uint8)
        cv2.ellipse(image,(80,120),(25,35),15,0,360,(230,230,230),-1)
        cv2.ellipse(image,(230,120),(32,24),-10,0,360,(245,245,245),-1)
        mask=segment(image,Config(method='multilevel',circularity=.5,solidity=.85))
        self.assertEqual(mask.shape,image.shape[:2])
        self.assertEqual(cv2.connectedComponents(mask.astype(np.uint8))[0]-1,2)
        self.assertTrue(mask[120,80] and mask[120,230])

    def test_ids_survive_detection_order_and_short_occlusion(self):
        tracker=Tracker(gate=100,max_missed=10)
        first=tracker.update([np.array([50,50]),np.array([200,50])],0)
        left,right=first[0].id,first[1].id
        reversed_order=tracker.update([np.array([198,50]),np.array([52,50])],1)
        self.assertEqual(reversed_order[0].id,right)
        self.assertEqual(reversed_order[1].id,left)
        tracker.update([],2)
        recovered=tracker.update([np.array([55,50]),np.array([195,50])],3)
        self.assertEqual(recovered[0].id,left); self.assertEqual(recovered[1].id,right)

if __name__=='__main__': unittest.main()
