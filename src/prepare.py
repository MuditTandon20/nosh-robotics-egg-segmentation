"""Download inputs; freeze independently inspected point prompts and scene groups."""
import argparse
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Normalized foreground points, specified by visual inspection before classical trials.
# These are localization prompts, never classical-method outputs or pixel labels.
POINTS = {
1:[(.49,.45)],2:[(.43,.33)],3:[(.47,.39)],4:[(.59,.32)],5:[(.50,.59)],6:[(.55,.20)],
10:[(.43,.59)],11:[(.61,.37)],12:[(.56,.49)],13:[(.61,.34)],14:[(.44,.48)],15:[(.50,.64)],16:[(.53,.45)],
17:[(.54,.56)],18:[(.53,.53)],19:[(.53,.37)],20:[(.47,.38)],21:[(.50,.43)],22:[(.68,.46)],24:[(.60,.43)],
25:[(.53,.49)],27:[(.48,.53)],28:[(.50,.28)],29:[(.68,.51)],30:[(.52,.62)],31:[(.50,.56)],32:[(.63,.53)],
33:[(.54,.45)],34:[(.47,.43)],35:[(.55,.57)],36:[(.45,.35)],37:[(.53,.71)],39:[(.68,.35)],40:[(.40,.63)],
41:[(.41,.59)],42:[(.35,.20)],43:[(.68,.50)],44:[(.47,.43)],45:[(.50,.27)],49:[(.45,.42)],
50:[(.46,.40)],52:[(.43,.31)],55:[(.68,.36)],62:[(.53,.60)],78:[(.49,.49)],95:[(.59,.28)],
100:[(.53,.52)],101:[(.44,.49)],102:[(.48,.38)],103:[(.53,.19),(.63,.48),(.66,.79)],
104:[(.49,.56)],105:[(.54,.33),(.72,.58),(.17,.73)],106:[(.50,.67)],107:[(.49,.52)],108:[(.49,.36)],
109:[(.50,.45)],110:[(.48,.44)],111:[(.55,.60)],112:[(.52,.51)],113:[(.65,.49),(.28,.68)],
114:[(.67,.50),(.23,.70)],115:[(.37,.34),(.52,.61)],116:[(.76,.73),(.31,.20)],117:[(.61,.42)],118:[(.56,.65)],119:[(.49,.65)],
120:[(.68,.59)],121:[(.24,.56),(.53,.76),(.86,.46)],122:[(.37,.39)],123:[(.51,.52)],124:[(.61,.40),(.40,.75)],125:[(.49,.48)],126:[(.45,.68)],
127:[(.52,.58)],128:[(.24,.70),(.79,.47)],129:[(.25,.70),(.75,.52)],130:[(.55,.72)],131:[(.51,.46)],
132:[(.36,.50),(.71,.66),(.86,.41)],133:[(.16,.22),(.20,.49),(.79,.52)],
134:[(.10,.76),(.50,.33),(.65,.57)],135:[(.59,.39),(.39,.75)],136:[(.55,.19),(.48,.48),(.51,.78)],138:[(.56,.63)],139:[(.35,.35)],
140:[(.49,.19),(.53,.48),(.63,.79)],141:[(.52,.61)],142:[(.26,.42),(.56,.54),(.26,.73)],
143:[(.38,.67)],144:[(.45,.34),(.32,.61),(.78,.53)],145:[(.62,.38),(.42,.74)],146:[(.50,.49)],147:[(.28,.72),(.70,.54)],
148:[(.30,.32),(.63,.51),(.21,.68)],149:[(.39,.63)],150:[(.47,.58)],151:[(.51,.45)],152:[(.46,.55)],
153:[(.48,.59)],154:[(.48,.43)],155:[(.42,.51)],156:[(.43,.40)],157:[(.48,.60)],158:[(.47,.31)],159:[(.32,.44)],
160:[(.55,.26)],161:[(.50,.52)],162:[(.41,.46)],163:[(.52,.42)],164:[(.55,.46)]}

SCENES = {
'hand':[1,2,25,102,151,152,155,157,158,160,161],
'striped_cloth':[3,4,5,6], 'grass':[10,11,12,13,49],
'granite':[14,27,36,41,50,78], 'patterned_fabric':[15,37,55,95],
'cable_fabric':[16,52,62], 'keyboard':[17,18,19,43],
'backlit_hand':[20,24,44], 'pink_bowl':[21,22,34,35,39,40],
'gray_floor':[28,45,146], 'brown_counter':[29,100,101,103,105,115,121,132,133,134,136,140,142,144,148],
'white_bag':[30,32,104,106,109,111,112,123,125,143],
'red_lid':[31], 'gray_surface':[33], 'egg_tray':[42,116,117],
'rice':[107,122,126], 'orange_container':[108,110,120,127,130,139,141],
'steel_bowl':[113,114,124,128,129,135,145,147,153,159],
'foil':[118,119,131,138], 'tile_blue':[149,150,156,163], 'woven_mat':[154,162,164]}

# Independent coarse boxes added during reference-mask QC, before final evaluation.
# Bounds are visual localization cues, not traced egg boundaries.
BOXES = {
20:[(.36,.22,.68,.52)],24:[(.35,.22,.83,.62)],40:[(.14,.50,.62,.83)],
116:[(.56,.48,.94,.88),(.05,.02,.40,.36)],117:[(.39,.14,.88,.54)],
121:[(.08,.35,.38,.72),(.37,.63,.67,.89),(.72,.32,.99,.61)],
123:[(.37,.32,.68,.67)],124:[(.42,.24,.77,.57),(.18,.57,.60,.93)],
128:[(.04,.51,.48,.84),(.56,.27,.97,.67)],129:[(.03,.50,.50,.89),(.56,.28,.95,.72)],
130:[(.24,.55,.83,.88)],134:[(.00,.61,.34,.90),(.36,.15,.66,.49),(.51,.43,.80,.72)],
135:[(.42,.24,.77,.57),(.18,.57,.60,.93)],142:[(.10,.25,.43,.63),(.33,.38,.80,.72),(.10,.60,.44,.92)],
144:[(.27,.13,.69,.52),(.10,.40,.55,.88),(.53,.31,.98,.73)],
145:[(.44,.23,.81,.57),(.21,.57,.64,.93)],160:[(.36,.12,.77,.45)]}

# Full-resolution audit corrected localization points that were ambiguous in thumbnails.
POINTS.update({20:[(.39,.34)],24:[(.46,.44)],116:[(.79,.68)],117:[(.58,.30)],
121:[(.17,.40),(.54,.63),(.86,.36)],123:[(.45,.34)],
128:[(.25,.49),(.78,.31)],129:[(.26,.52),(.72,.36)],130:[(.51,.56)],
134:[(.21,.68),(.72,.28),(.84,.52)],142:[(.40,.28),(.30,.55),(.64,.40)],
144:[(.44,.19),(.32,.43),(.77,.33)],160:[(.45,.43)]})
BOXES.update({20:[(.23,.20,.56,.49)],24:[(.23,.28,.73,.64)],116:[(.60,.50,.95,.85)],
117:[(.38,.12,.84,.48)],121:[(.03,.24,.32,.56),(.38,.46,.71,.78),(.72,.20,.99,.52)],
123:[(.30,.16,.62,.51)],128:[(.02,.35,.48,.65),(.55,.14,.98,.50)],
129:[(.02,.34,.50,.72),(.50,.18,.95,.55)],130:[(.24,.35,.79,.74)],
134:[(.00,.52,.48,.83),(.57,.11,.87,.45),(.68,.36,.99,.68)],
142:[(.23,.06,.58,.46),(.10,.39,.49,.74),(.40,.23,.86,.57)],
144:[(.22,.01,.66,.40),(.05,.24,.59,.69),(.52,.15,.99,.52)],
160:[(.25,.26,.66,.60)]})

def prepare(download=False):
    if download:
        import gdown
        import requests
        (ROOT/'data').mkdir(exist_ok=True)
        gdown.download(id='1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv', output=str(ROOT/'data/dataset.zip'))
        with zipfile.ZipFile(ROOT/'data/dataset.zip') as archive:
            archive.extractall(ROOT/'data/raw')
        (ROOT/'models').mkdir(exist_ok=True)
        response=requests.get('https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt',timeout=180)
        response.raise_for_status()
        (ROOT/'models/mobile_sam.pt').write_bytes(response.content)
    files=sorted((ROOT/'data/raw').rglob('*.jpg'))
    assert len(files)==110, f'Expected 110 images, found {len(files)}'
    groups=list(SCENES)
    random.Random(2026).shuffle(groups)
    # Exact subset sum on whole groups: 22 eval images, 88 development images.
    subsets={0:[]}
    for group in groups:
        for count, selected in list(subsets.items()):
            n=count+len(SCENES[group])
            if n<=22 and n not in subsets: subsets[n]=selected+[group]
    eval_groups=subsets[22]
    records=[]
    for path in files:
        number=int(re.search(r'not_damaged_(\d+)',path.name)[1])
        group=next(g for g,ids in SCENES.items() if number in ids)
        records.append(dict(id=number,path=path.relative_to(ROOT).as_posix(),group=group,
                            split='eval' if group in eval_groups else 'dev',points=POINTS[number],boxes=BOXES.get(number),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    records.sort(key=lambda r:r['id'])
    (ROOT/'data/manifest.json').write_text(json.dumps(records,indent=2))
    print('Prepared',len(records),'images; eval groups:',eval_groups)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--download',action='store_true')
    prepare(parser.parse_args().download)
