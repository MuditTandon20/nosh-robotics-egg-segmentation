"""Image-only segmentation. No prompts, reference masks or image IDs are inputs."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from scipy import ndimage

@dataclass(frozen=True)
class Config:
    method: str='otsu'
    saturation: int=85
    value: int=110
    min_area: float=.003
    max_area: float=.40
    circularity: float=.45
    solidity: float=.82
    close: int=5
    grabcut: int=0

TRIALS=[
    ('01_otsu',Config(), 'Bright egg against darker background; establish a simple baseline.'),
    ('02_low_saturation',Config(method='hsv',saturation=70,value=130), 'Reject colored backgrounds using low saturation and high brightness.'),
    ('03_relaxed_hsv',Config(method='hsv',saturation=100,value=95), 'Recover shadowed egg regions by relaxing color thresholds.'),
    ('04_shape_filter',Config(method='hsv',saturation=85,value=110,circularity=.60,solidity=.88), 'Reject irregular background fragments with stronger shape constraints.'),
    ('05_lab_otsu',Config(method='lab',value=95,saturation=95), 'Use illumination thresholding in Lab and suppress chromatic pixels.'),
    ('06_grabcut',Config(method='hsv',saturation=85,value=110,grabcut=3), 'Refine candidate boundaries with foreground/background color models.'),
    ('07_close_large',Config(method='hsv',saturation=85,value=110,close=11), 'Bridge egg highlights and small cracks; risk merging background.'),
    ('08_local_threshold',Config(method='local',saturation=85,value=100), 'Try local contrast to handle uneven illumination; may amplify textured backgrounds.'),
    ('09_multilevel_ovals',Config(method='multilevel',circularity=.65,solidity=.90), 'Development failures show that one threshold merges eggs with pale backgrounds. Search multiple thresholds and rank oval candidates.'),
    ('10_multilevel_relaxed',Config(method='multilevel',circularity=.50,solidity=.85), 'Recover less regular egg contours; test whether increased recall outweighs background false positives.'),
]

def multilevel(work,config):
    gray=cv2.cvtColor(cv2.GaussianBlur(work,(7,7),0),cv2.COLOR_BGR2GRAY)
    saturation=cv2.cvtColor(work,cv2.COLOR_BGR2HSV)[:,:,1]
    candidates=[]
    for threshold in range(55,241,10):
        mask=(gray>threshold).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            if not .004*mask.size<area<.45*mask.size or len(contour)<5: continue
            perimeter=cv2.arcLength(contour,True)
            circularity=4*np.pi*area/max(perimeter**2,1)
            solidity=area/max(cv2.contourArea(cv2.convexHull(contour)),1)
            if circularity<config.circularity or solidity<config.solidity: continue
            ellipse=cv2.fitEllipse(contour); (cx,cy),(a,b),angle=ellipse
            if min(a,b)<12 or max(a,b)/min(a,b)>2.3: continue
            if cx<.04*384 or cx>.96*384 or cy<.04*384 or cy>.96*384: continue
            filled=np.zeros(mask.shape,np.uint8); cv2.drawContours(filled,[contour],-1,1,-1)
            ellipse_mask=np.zeros(mask.shape,np.uint8); cv2.ellipse(ellipse_mask,ellipse,1,-1)
            overlap=(filled&ellipse_mask).sum()/max((filled|ellipse_mask).sum(),1)
            if overlap<.80: continue
            inside=filled>0
            sat=float(saturation[inside].mean())
            if sat>110: continue
            ring=(cv2.dilate(filled,np.ones((13,13),np.uint8))>0)&~inside
            contrast=float(gray[inside].mean()-gray[ring].mean()) if ring.any() else 0
            # Favor smooth closed neutral ovals; mild brightness and contrast preferences.
            score=overlap*.40+solidity*.15+circularity*.15+(1-sat/255)*.15+min(max(contrast,0)/80,1)*.15
            candidates.append((score,filled))
    result=np.zeros(gray.shape,np.uint8)
    candidates.sort(key=lambda item:item[0],reverse=True)
    if not candidates: return result
    accepted=[]
    for score,mask in candidates:
        if score<max(.72,candidates[0][0]-.09): continue
        if any((mask&old).sum()/max(min(mask.sum(),old.sum()),1)>.25 for old in accepted): continue
        accepted.append(mask); result|=mask
        if len(accepted)>=4: break
    return result

def segment(image, config=Config()):
    h,w=image.shape[:2]
    # Fixed working resolution makes morphology scale consistent.
    work=cv2.resize(image,(384,384))
    if config.method=='multilevel':
        result=multilevel(work,config)
        return cv2.resize(result,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
    smooth=cv2.GaussianBlur(work,(5,5),0)
    hsv=cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(smooth,cv2.COLOR_BGR2GRAY)
    if config.method=='otsu':
        _,seed=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif config.method=='lab':
        light=cv2.cvtColor(smooth,cv2.COLOR_BGR2LAB)[:,:,0]
        threshold,_=cv2.threshold(light,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        seed=((light>max(config.value,threshold))&(hsv[:,:,1]<config.saturation)).astype(np.uint8)*255
    elif config.method=='local':
        seed=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,-3)
        seed[(hsv[:,:,1]>config.saturation)|(hsv[:,:,2]<config.value)]=0
    else:
        seed=((hsv[:,:,1]<config.saturation)&(hsv[:,:,2]>config.value)).astype(np.uint8)*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(config.close,config.close))
    seed=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,kernel)
    seed=cv2.morphologyEx(seed,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(seed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=np.zeros(seed.shape,np.uint8)
    for contour in contours:
        area=cv2.contourArea(contour); perimeter=cv2.arcLength(contour,True)
        if not config.min_area*seed.size<area<config.max_area*seed.size: continue
        circularity=4*np.pi*area/max(perimeter**2,1)
        solidity=area/max(cv2.contourArea(cv2.convexHull(contour)),1)
        x,y,bw,bh=cv2.boundingRect(contour)
        if circularity<config.circularity or solidity<config.solidity or max(bw,bh)/max(min(bw,bh),1)>2.5: continue
        cv2.drawContours(result,[contour],-1,255,-1)
    if config.grabcut and np.any(result) and np.any(result==0):
        sure=cv2.erode(result,np.ones((7,7),np.uint8))>0
        possible=cv2.dilate(result,np.ones((21,21),np.uint8))>0
        labels=np.where(possible,cv2.GC_PR_BGD,cv2.GC_BGD).astype(np.uint8)
        labels[result>0]=cv2.GC_PR_FGD; labels[sure]=cv2.GC_FGD
        if sure.any():
            cv2.grabCut(work,labels,None,np.zeros((1,65)),np.zeros((1,65)),config.grabcut,cv2.GC_INIT_WITH_MASK)
            result=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)*255
    result=ndimage.binary_fill_holes(result>0).astype(np.uint8)
    return cv2.resize(result,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)

def configuration_dict(config):
    return asdict(config)
