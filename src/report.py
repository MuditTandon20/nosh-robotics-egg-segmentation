"""Build measured plots, qualitative comparisons, Markdown and PDF report."""
import csv
import html
import json
import platform
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from .prepare import ROOT

def read(name): return json.loads((ROOT/name).read_text())

def build():
    folder=ROOT/'outputs/figures'; folder.mkdir(parents=True,exist_ok=True)
    summary=read('outputs/summary.json'); experiments=read('outputs/experiments.json')
    tracking=read('outputs/tracking_summary.json'); records=read('data/manifest.json')
    publication=read('outputs/publication.json')
    rows=list(csv.DictReader((ROOT/'outputs/per_image_metrics.csv').open()))
    for row in rows:
        row['id']=int(row['id'])
        for key in ['iou','dice','precision','recall','boundary_f1']: row[key]=float(row[key])
    evaluation=[r for r in rows if r['split']=='eval']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    axes[0].barh([e['trial'] for e in experiments],[e['development']['iou'] for e in experiments],color=['#0a887e' if e['decision']=='kept' else '#92a9bc' for e in experiments])
    axes[0].invert_yaxis(); axes[0].set_xlabel('Development mean IoU'); axes[0].set_xlim(0,1)
    axes[1].hist([r['iou'] for r in evaluation],bins=np.linspace(0,1,11),color='#0a887e',edgecolor='white')
    axes[1].set_xlabel('Evaluation image IoU'); axes[1].set_ylabel('Images'); axes[1].set_xlim(0,1)
    fig.tight_layout(); fig.savefig(folder/'experiments.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    by_scene=summary['eval_by_scene']
    axes[0].bar(range(len(by_scene)),[s['iou'] for s in by_scene.values()],color='#0a887e')
    axes[0].set_xticks(range(len(by_scene)),[g.replace('_','\n') for g in by_scene]); axes[0].set_ylim(0,1); axes[0].set_ylabel('Mean IoU')
    axes[1].scatter([r['recall'] for r in evaluation],[r['precision'] for r in evaluation],c=[r['iou'] for r in evaluation],cmap='viridis',vmin=0,vmax=1,s=50)
    axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(-.03,1.03),ylim=(-.03,1.03))
    fig.tight_layout(); fig.savefig(folder/'errors.png',dpi=180); plt.close(fig)
    worst=sorted(evaluation,key=lambda r:r['iou'])[:4]
    best=max(evaluation,key=lambda r:r['iou'])
    selected=worst+[best]
    qualitative=[]
    for row in selected:
        record=next(r for r in records if r['id']==row['id'])
        image=cv2.cvtColor(cv2.imread(str(ROOT/record['path'])),cv2.COLOR_BGR2RGB)
        reference=cv2.imread(str(ROOT/f"outputs/reference/{row['id']:03d}.png"),-1)>0
        prediction=cv2.imread(str(ROOT/f"outputs/predictions/{row['id']:03d}.png"),-1)>0
        error=np.zeros_like(image); error[reference&prediction]=[30,160,130]
        error[reference&~prediction]=[235,150,25]; error[~reference&prediction]=[220,55,85]
        fig,axes=plt.subplots(1,4,figsize=(11,3))
        for ax,data,title in zip(axes,[image,reference,prediction,error],['Image','MobileSAM reference','Classical prediction','Error map']):
            ax.imshow(data,cmap='gray' if data.ndim==2 else None); ax.set_title(title); ax.axis('off')
        fig.suptitle(f"ID {row['id']} | {record['group']} | IoU {row['iou']:.3f} | recall {row['recall']:.3f}",fontsize=11)
        fig.tight_layout(); path=folder/f"sample_{row['id']:03d}.png"; fig.savefig(path,dpi=180); plt.close(fig)
        if row['recall']<.2: note='Most of the reference egg is missed. The contour search fails to find a sufficiently closed oval at any threshold.'
        elif row['precision']<.65: note='Background is included alongside the egg. Similar brightness and rounded background fragments defeat the color and shape priors.'
        else: note='The recovered region is too small or has a displaced boundary. Highlights and shadows isolate part of the shell instead of its whole silhouette.'
        if row is best: note='Successful example: a neutral shell is separated from its background at one of the searched thresholds, producing a closed oval.'
        qualitative.append((path,note,row))
    # Audit exact hashes and perceptual similarity across splits; dHash is a flag, not proof.
    hashes={}
    for r in records:
        gray=cv2.imread(str(ROOT/r['path']),0); small=cv2.resize(gray,(32,32)).astype(float)
        coeff=dctn(small,norm='ortho')[:8,:8].ravel()[1:]
        hashes[r['id']]=coeff>np.median(coeff)
    pairs=sorted((int(np.count_nonzero(hashes[a['id']]!=hashes[b['id']])),a['id'],b['id'])
                 for a in records if a['split']=='dev' for b in records if b['split']=='eval')
    audit={'exact_cross_split_duplicates':len({r['sha256'] for r in records if r['split']=='dev'}&{r['sha256'] for r in records if r['split']=='eval'}),
           'minimum_cross_split_phash_distance':pairs[0][0], 'closest_cross_split_pairs':pairs[:10],
           'caveat':'63-bit low-frequency pHash is a diagnostic, not a guarantee of scene independence.'}
    (ROOT/'outputs/leakage_audit.json').write_text(json.dumps(audit,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(11,4.8))
    for ax,index in zip(axes.flat,[288,480,576,768,864,960]):
        path=ROOT/f'outputs/tracking_{index:04d}.jpg'
        ax.imshow(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB));ax.set_title(f'{index/24:.0f} seconds');ax.axis('off')
    fig.tight_layout();fig.savefig(folder/'tracking.png',dpi=180);plt.close(fig)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#25364a')))
    styles.add(ParagraphStyle(name='Compact',parent=styles['BodyCustom'],fontSize=9,leading=12,spaceAfter=6))
    styles['Title'].textColor=colors.HexColor('#13334c'); styles['Heading1'].textColor=colors.HexColor('#087c77')
    styles['Heading2'].textColor=colors.HexColor('#13334c')
    story=[]; markdown=[]
    def title(text): story.append(Paragraph(html.escape(text),styles['Title']));markdown.append('# '+text+'\n')
    def heading(text): story.append(Paragraph(html.escape(text),styles['Heading1']));markdown.append('\n## '+text+'\n')
    def para(text,compact=False): story.append(Paragraph(text,styles['Compact' if compact else 'BodyCustom']));markdown.append(text.replace('<b>','').replace('</b>','')+'\n')
    def picture(path,width=7*inch):
        from PIL import Image as PILImage
        w,h=PILImage.open(path).size
        story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,8))
        markdown.append(f'![{path.stem}]({path.relative_to(ROOT).as_posix()})\n')
    def table(data,widths=None):
        converted=[[Paragraph(html.escape(str(c)),styles['BodyCustom']) for c in row] for row in data]
        t=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6f2f1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#087c77')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,10))
        markdown.append('| '+' | '.join(str(c) for c in data[0])+' |\n')
        markdown.append('| '+' | '.join('---' for _ in data[0])+' |\n')
        markdown.extend(['| '+' | '.join(str(c) for c in row)+' |\n' for row in data[1:]])
    def page(): story.append(PageBreak())
    title('Egg segmentation and individual tracking')
    para('Nosh Robotics CV Intern Assignment | 28 October 2026')
    para('This submission generates pretrained reference masks for all 110 images, develops an image-only classical segmenter, compares ten measured development trials, and processes all 1,440 frames of the supplied 60-second video.')
    heading('Measured result')
    table([['Metric','Development (88)','Evaluation (22)']]+[[label,f"{summary['dev'][key]:.3f}",f"{summary['eval'][key]:.3f}"] for label,key in [('Mean IoU','iou'),('Mean Dice','dice'),('Mean precision','precision'),('Mean recall','recall'),('Boundary F1 (3 px)','boundary_f1'),('Micro IoU','micro_iou')]], [230,130,130])
    ci=summary['eval']['iou_cluster_bootstrap_95ci']
    para(f"Evaluation mean IoU is <b>{summary['eval']['iou']:.3f}</b>; the scene-cluster bootstrap interval is {ci[0]:.3f}-{ci[1]:.3f}. Only four evaluation scene groups contribute, so uncertainty is substantial. Recall trails precision: missing egg regions are the dominant error. This method does not closely match MobileSAM on all backgrounds.")
    para(f"The tracking output contains {tracking['frames']} frames at {tracking['fps']:.0f} fps and emits {tracking['total_ids']} track IDs. It is silent and retains the full source duration. ID count is a diagnostic, not a tracking-accuracy score.")
    para('Public repository: <link href="'+publication['github_url']+'">'+publication['github_url']+'</link>')
    drive=publication.get('drive_url','')
    para('Tracking video on Google Drive: <link href="'+drive+'">Open tracking video</link>. Public access verification: '+str(publication.get('drive_public_status','pending'))+'. A downloadable copy is also included in the public repository at outputs/egg_tracking.mp4.')
    heading('Scope and honest limits')
    para('Reference masks are pretrained-model pseudo-ground truth, not independent human pixel annotations. Difficult submerged eggs remain uncertain. An early exploratory run produced evaluation metrics before reference QC and the final candidate method were finished; the final split is scene-disjoint, but it must not be described as a pristine blind test. No parameters were chosen by maximizing the final evaluation result.')
    page();heading('1. Data, leakage reduction and reference masks')
    para('The assignment links the egg-vzhtr version-1 Roboflow dataset and an alternative ZIP. The ZIP contains 110 JPEGs distributed across train, test and valid folders, including two images in an Unlabeled folder. All images are included. Original folder labels do not provide pixel masks. The bundled dataset README declares CC BY 4.0 and 640 x 640 stretched preprocessing; there are no exported augmentations.')
    para('A visual scene inventory assigns the 110 images to 21 conservative groups. Related views of hands, the same counter, bowl, fabric, keyboard and other setups stay together. A seed-2026 subset-sum allocation selects whole groups totaling exactly 22 images; 88 remain for development. Evaluation groups are cable_fabric (3), grass (5), striped_cloth (4), and steel_bowl (10). Source images and independent prompts are recorded in data/manifest.json.')
    para(f"Exact cross-split duplicates: {audit['exact_cross_split_duplicates']}. Minimum cross-split 63-bit perceptual-hash Hamming distance: {audit['minimum_cross_split_phash_distance']}. The nearest pairs are recorded in outputs/leakage_audit.json. Scene grouping and hashes reduce leakage; camera/session metadata is unavailable and scene independence cannot be proven.")
    para('Task 1 uses the official pretrained MobileSAM vit_t checkpoint on CPU, without fine-tuning. MobileSAM preserves SAM\'s promptable mask-decoder interface with a smaller encoder. Foreground points are specified independently by visual inspection. Other egg centers become negative points to keep instance masks separate. Selected ambiguous images receive independent coarse localization boxes. This is a prompted pretrained pipeline, not autonomous semantic egg detection.')
    para('For each prompt, inference returns three masks and their model quality scores; the highest-scoring mask is retained. Instance labels are saved as uint16 PNGs, with 0 as background. Their union is the semantic reference used in evaluation. Raw quality scores can exceed 1 and must not be interpreted as calibrated probabilities. Model SHA-256, device, timing and scores are in outputs/reference_audit.json.')
    para('A thumbnail audit was followed by full-resolution review of suspicious examples. Points were corrected where they landed on bars, shadows or pot rims; coarse boxes were added to reduce background leakage. These corrections use image appearance, not classical predictions. No model masks were replaced with traced manual pixel masks. Images 124, 135 and 145 show eggs through boiling water; bubbles and weak boundaries make their model references especially uncertain. Their scores measure agreement with these references rather than objective segmentation accuracy.')
    para('Reproducibility: data/manifest.json fixes prompts, boxes, groups and image SHA-256 hashes. Re-running src.prepare reproduces the same manifest and split. Checkpoint provenance and references appear at the end of this report.')
    page();heading('2. Classical method and experiment design')
    para('All classical runs use image pixels alone. There is no access to prompts, source IDs, scene labels or reference masks during inference. Images are resized to 384 x 384 for consistent morphology. The first eight trials test grayscale Otsu, HSV brightness/saturation limits, shape filtering, Lab thresholding, GrabCut refinement, larger closing, and local adaptive thresholding.')
    para('The development failures of fixed thresholds motivated a multilevel method. It blurs grayscale with a 7 x 7 kernel, searches thresholds 55 through 235 in steps of 10, closes with a 5 x 5 kernel, and examines closed contours. Candidates must occupy 0.4%-45% of the working image, have at least five contour points, fitted minor diameter at least 12 pixels, axis ratio at most 2.3, center inside the inner 92% of the frame, ellipse IoU at least 0.80, and mean HSV saturation at most 110.')
    para('Candidate score = 0.40 x ellipse IoU + 0.15 x solidity + 0.15 x circularity + 0.15 x (1 - mean saturation / 255) + 0.15 x clipped inside-versus-ring contrast / 80. The ring is generated by a 13 x 13 dilation. Accept scores at least max(0.72, best score - 0.09). Suppress candidates overlapping an accepted region by more than 25% of the smaller area; retain at most four. The selected trial uses circularity >= 0.50 and solidity >= 0.85. The union is resized back with nearest-neighbor interpolation.')
    para('Trials 01-08 were a predefined ablation batch. Trials 09-10 were added after the development failures above. The winning configuration is selected solely by development macro IoU; there is no training step. The full trial records include exact parameters, motivation, all development metrics and keep/discard reasons in outputs/experiments.json. Each trial below is a real completed run.')
    table([['Trial','Dev IoU','Dev Dice','Decision']]+[[e['trial'],f"{e['development']['iou']:.3f}",f"{e['development']['dice']:.3f}",e['decision']] for e in experiments],[220,85,85,100])
    page();heading('Experiment log: parameters and decisions')
    for e in experiments:
        p=e['parameters']
        para('<b>'+e['trial']+'</b> — '+e['reason'],compact=True)
        para('Parameters: '+', '.join(f'{k}={v}' for k,v in p.items())+'. Development IoU '+f"{e['development']['iou']:.3f}"+', Dice '+f"{e['development']['dice']:.3f}"+'. '+e['decision'].capitalize()+': '+e['decision_reason']+'.',compact=True)
    para('For non-multilevel methods, minimum/maximum area are fractions of the 384 x 384 frame, close is kernel width, value/saturation use OpenCV 0-255 channels, and grabcut counts iterations. The multilevel branches use the explicitly stated area, blur and kernel settings above; Config fields for HSV thresholds, close and GrabCut do not affect those branches.',compact=True)
    page();heading('3. Quantitative evaluation and interpretation')
    para('For foreground pixels, TP is intersection, FP is prediction outside the reference, and FN is reference missed by prediction. IoU = TP/(TP+FP+FN); Dice = 2TP/(2TP+FP+FN); precision = TP/(TP+FP); recall = TP/(TP+FN). Reported primary scores are image means. Micro IoU pools pixels across images and therefore weights larger reference regions more heavily. Empty reference and prediction score 1; an empty prediction against a nonempty reference scores 0.')
    para('Boundary F1 compares boundaries obtained by binary erosion, with matches within three pixels at original 640 x 640 resolution. Bootstrap confidence bounds resample whole scene groups with replacement, 2,000 times using seed 2026; they are conditional on this small split and model-generated references. Per-scene single-group confidence bounds are degenerate and should not be interpreted as uncertainty estimates.')
    picture(folder/'experiments.png');picture(folder/'errors.png')
    para('The broad gap between precision and recall indicates conservative selection: pale backgrounds merge with shells, while shadows and textures produce fragmented or non-oval regions that fail the shape filters. Lower boundary F1 shows that even detected eggs often have inaccurate silhouettes. Evaluation IoU exceeds development IoU here, which reflects scene composition rather than evidence of better generalization.')
    para(f"Classical development inference averages {summary['dev']['seconds']*1000:.1f} ms/image and evaluation inference {summary['eval']['seconds']*1000:.1f} ms/image on this run, excluding image loading and metric computation. Timing is a single-run measurement affected by concurrent work; no throughput benchmark is claimed.")
    page();heading('Qualitative failures and a successful example')
    para('Examples are selected mechanically: the four lowest-IoU evaluation images and the highest-IoU image. Error colors: teal = intersection, amber = missed reference foreground, red = extra prediction. These figures expose errors rather than selecting only successful samples.')
    for path,note,row in qualitative[:2]: picture(path);para(note)
    page();heading('Further qualitative examples')
    for path,note,row in qualitative[2:]: picture(path);para(note)
    page();heading('4. Video tracking')
    para('The supplied YouTube clip is 60 seconds, 640 x 360 at 24 fps. Every decoded frame is processed and written. Intro/outro frames are retained, and no detections are fabricated during occlusion. Detection adapts the classical component pipeline to the tabletop domain: HSV saturation < 75, value > 170, 3 x 3 closing, area 0.6%-3.5%, circularity >= 0.55 and solidity >= 0.82. Masks outside x=2%-94% and y=42%-90% are suppressed to exclude the shirt, title and logo. This video-specific tuning does not change the evaluated still-image method.')
    para('Contours supply centroids to a constant-velocity Kalman filter. Hungarian assignment minimizes centroid distance under a 256-pixel gate. Velocity is damped and zeroed after three missed frames to limit drift while a hand covers an egg. Tracks survive up to 192 missed frames (8 seconds), chosen after visual inspection of the clip\'s long hand occlusions. IDs are associated with detections only; missed tracks are not drawn as visible eggs. Full detections and IDs are exported to outputs/tracks.csv.')
    para(f"Run summary: {tracking['frames']} frames, {tracking['total_ids']} total IDs, detection-count histogram {tracking['detection_count_histogram']}. The threshold/shape detector was refined to remove small white text/reflection fragments; tracking retention was increased after inspecting identity fragmentation under hands. These are video-specific experiments, not held-out video validation.")
    picture(folder/'tracking.png')
    para('Limitations: a long retention window can associate the wrong egg after a full occlusion, and centroid assignment does not establish identity through crossing, cuts or exits. Two IDs alone do not prove zero switches. There are no manually annotated trajectories or masks for this video, so MOTA, IDF1 and HOTA are not reported. The output is a visual demonstration with track logs, not a quantitatively validated tracker.')
    page();heading('Improvements, reproducibility and AI disclosure')
    para('Highest-priority improvements: (1) manually audit and correct an independent subset of model references, especially submerged eggs; (2) collect more genuinely independent scenes and reserve a new untouched evaluation set; (3) use edge continuity or marker-controlled watershed to recover shells merged into pale backgrounds and separate touching eggs; (4) add texture and local background models to reject smooth background ovals; (5) annotate a short video interval and measure detection recall, fragmentation and identity switches before tuning re-identification.')
    para('The code is divided into preparation, pretrained inference, classical segmentation, evaluation, tracking and report generation modules. README.md gives exact setup and commands. Artifacts include all 110 reference masks, all 110 final prediction masks, per-image CSV metrics, full experiment JSON, selected configuration, scene/hash audit, plots, track CSV and MP4. Raw data and weights are fetched separately and excluded from Git.')
    para('Validation performed: known-overlap metric arithmetic and empty-mask conventions; exactly 88/22 image counts and scene separation; image-hash uniqueness and reference-mask completeness; recovery of two synthetic eggs; and identity stability under reordered detections and a short synthetic occlusion. A decoded-frame validation also checks the output video duration and frame count. These checks do not replace manual scene or reference annotation.')
    para('AI disclosure: OpenAI Codex assisted with implementing the Python pipeline, inspecting image thumbnails, specifying/correcting point and box prompts, proposing classical trials, running experiments, diagnosing failures, preparing visualizations and writing this report. The numerical results are generated from executed code, not invented. The candidate should study and reproduce the implementation and be ready to explain and modify it in the interview; this report does not assert independent candidate authorship or understanding.')
    para('Experimental integrity note: an early exploratory run generated evaluation scores before final reference review. Later methods were motivated by development failures, and final ranking uses development results only. Because evaluation outputs had already been exposed during development, the final scene-disjoint evaluation is exploratory rather than a fully blind confirmation. This is a limitation to resolve with new data.')
    heading('Sources and attribution')
    for label,url in [
        ('Assignment dataset / Roboflow, CC BY 4.0','https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1'),
        ('Alternative dataset archive','https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view'),
        ('SAM: official implementation and promptable segmentation','https://github.com/facebookresearch/segment-anything'),
        ('MobileSAM: official code and checkpoint, Apache 2.0','https://github.com/ChaoningZhang/MobileSAM'),
        ('Source video: Mocomi, Raw and Boiled Egg Spinning','https://www.youtube.com/watch?v=Y4JyV94y2_A')]:
        para(label+': <link href="'+url+'">'+url+'</link>')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607487'))
        canvas.drawString(36,22,'Nosh Robotics CV assignment | measured reproducible experiments')
        canvas.drawRightString(576,22,str(doc.page))
    document=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(612,792),leftMargin=36,rightMargin=36,topMargin=36,bottomMargin=38)
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'REPORT.md').write_text('\n'.join(markdown),encoding='utf-8')
    print('Built report.pdf and REPORT.md')

if __name__=='__main__': build()
