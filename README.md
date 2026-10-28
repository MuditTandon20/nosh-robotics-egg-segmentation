# Egg segmentation and tracking — Nosh Robotics

An executed CV assignment with pretrained MobileSAM reference masks, an image-only classical segmenter, ten development experiments, quantitative/qualitative analysis, and individual egg tracking.

- [PDF report](report.pdf) and [readable report](REPORT.md)
- [Tracking video — download the MP4](outputs/egg_tracking.mp4)
- [Google Drive video](https://drive.google.com/file/d/1XDrUo4D_BzqZPXqDqBrPhjaMHzaO1i-r/view?usp=drivesdk) — **private at the owner's request**. The assignment requests a public Drive link; this remaining sharing requirement is not satisfied while it is private. The public repository includes the same video.
- [Experiment log](outputs/experiments.json), [per-image metrics](outputs/per_image_metrics.csv), [summary](outputs/summary.json), [track logs](outputs/tracks.csv)

## Results

All 110 images are processed. A scene-group split allocates 88 images to development and 22 to evaluation. Selected method: threshold sweep + oval candidate scoring + overlap suppression.

| Image-mean metric | Development | Evaluation |
|---|---:|---:|
| IoU | 0.456 | 0.515 |
| Dice | 0.549 | 0.617 |
| Precision | 0.748 | 0.851 |
| Recall | 0.475 | 0.531 |
| Boundary F1, 3 pixels | 0.273 | 0.307 |

The full 60-second video has 1,440 frames at 24 fps. The final run emits two track IDs. This does **not** establish zero identity switches; no video trajectory annotations exist. The output is silent.

Reference masks are model-generated pseudo-ground truth. Localization prompts were visually specified, independently of the classical method; Task 1 is prompted inference, not autonomous egg detection. Water/bubble cases remain uncertain. An exploratory run exposed evaluation scores before the final reference audit, so the final scene-disjoint evaluation is **not a pristine blind test**. Parameters are ranked using development results. See the report for limitations and unsuccessful trials.

## Setup

Install Git and Python 3.10 or 3.11. Run commands from the repository root. CPU inference is supported; no API keys, paid services or GPU are needed.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. The requirements pin OpenCV 4.11 for NumPy 1.x compatibility. Initial exploration used OpenCV 5.0.0.93; final classical experiments, tracking and reports were rerun with the pinned version. The recorded environment is in `outputs/environment.json`. For CUDA, install a compatible PyTorch build according to the official PyTorch installation instructions. CPU inference for all reference masks takes roughly two minutes on the tested machine, excluding installation/downloads.

## Reproduce

```powershell
# Fetch the assignment's exact 4.39 MB dataset ZIP and ~40 MB MobileSAM checkpoint.
python -m src.prepare --download

# Generate one uint16 instance-label PNG per image; 0 means background.
python -m src.pretrained

# Run all ten dev trials, select by dev IoU, then evaluate the frozen winner.
python -m src.evaluate

# Download the exact source clip. YouTube availability varies.
python -m yt_dlp --js-runtimes node --extractor-args "youtube:player_client=android,web" -f "best[height<=480][ext=mp4]/best[height<=480]" -o "data/source_video.%(ext)s" "https://www.youtube.com/watch?v=Y4JyV94y2_A"

# Process all frames with the video-domain adaptation and Kalman/Hungarian tracker.
python -m src.track --input data/source_video.mp4 --output outputs/egg_tracking.mp4

# Rebuild figures and PDF from measured outputs.
python -m src.report

# Verify metric arithmetic, split integrity, synthetic segmentation and ID association.
python -m unittest discover -s tests -v

# Decode the entire output video and validate every mask and metric row.
python -m src.validate
```

If YouTube throttles downloading, obtain the same clip through an authorized download and pass its local path with `--input`. Node.js is optional except for the downloader's JavaScript support. No browser cookies are bundled or required by the code.

For development only, `python -m src.evaluate --dev-only` runs trials without evaluating the eval split. Model localization prompts and scene groups are defined in `src/prepare.py` and frozen in `data/manifest.json`. Re-running preparation does not infer prompts from classical outputs. A full reference run should precede evaluation; missing input files cause an error rather than a partial score.

## Layout

```text
src/prepare.py       dataset download, independent prompts/boxes, exact grouped split
src/pretrained.py    pretrained MobileSAM inference and checkpoint audit
src/classical.py     pixel-only classical segmentation and trial configurations
src/evaluate.py      metrics, dev selection, eval, grouped bootstrap
src/track.py         tabletop adaptation, Kalman filters, Hungarian association
src/report.py        plots, qualitative failure panels, Markdown/PDF report
tests/              four meaningful pipeline checks
data/manifest.json  image hashes, prompts, boxes, groups and split labels
outputs/reference/ uint16 pretrained instance masks (110 PNGs)
outputs/predictions/ final classical binary masks (110 PNGs)
outputs/figures/    publication/report figures
outputs/           experiment logs, metrics, track CSV and final MP4
report.pdf          submission report
```

The checked-in measured artifacts let reviewers inspect results without rerunning inference. Raw images, original ZIP, source video and weights are excluded from Git. The public annotated output retains the source attribution and logo. Dataset attribution: [Roboflow egg-vzhtr v1](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1), provided by a Roboflow user under CC BY 4.0 according to its bundled README. MobileSAM code/checkpoint: [official repository](https://github.com/ChaoningZhang/MobileSAM), Apache 2.0. Source clip: [Mocomi — Raw and Boiled Egg Spinning](https://www.youtube.com/watch?v=Y4JyV94y2_A). Third-party media and weights retain their original licenses.

## Understanding the implementation

The classical final method searches grayscale thresholds rather than assuming one brightness cutoff works everywhere. At each threshold it scores smooth, low-saturation, approximately oval contours using ellipse agreement, solidity, circularity and local contrast. Overlap suppression avoids selecting the same egg at multiple thresholds. This is transparent and fast, but cannot recognize an egg when a pale background joins its boundary or when the shell is heavily occluded. It has no semantic learned features.

The tracker predicts centroids with a constant-velocity Kalman filter and associates observations using globally optimal Hungarian assignment with a distance gate. Missed tracks are retained through hand occlusions but are not drawn as visible detections. A long retention period can preserve an ID or cause an incorrect reassociation; manual trajectory annotations would be needed to quantify that tradeoff.

## AI assistance

OpenAI Codex assisted with implementation, image inspection and localization prompts, experiment proposals/execution, debugging, visualizations and report writing. All reported metrics come from executed code. This does not assert independent candidate authorship or understanding. Review the methods and rerun the experiments before presenting or discussing the submission in an interview.
