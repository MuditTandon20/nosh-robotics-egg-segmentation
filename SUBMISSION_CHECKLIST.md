# Assignment submission checklist

Public repository: https://github.com/MuditTandon20/nosh-robotics-egg-segmentation

Report: [report.pdf](report.pdf)

Public Google Drive tracking output: https://drive.google.com/file/d/1XDrUo4D_BzqZPXqDqBrPhjaMHzaO1i-r/view

Drive access is verified as `type=anyone`, `role=reader`. No sign-in or invitation is required to view with the link.

| Assignment requirement | Deliverable / evidence |
|---|---|
| Task 1: pretrained segmentation of every dataset image | 110 uint16 instance masks in [outputs/reference](outputs/reference); [inference code](src/pretrained.py); [model audit](outputs/reference_audit.json) |
| Task 2: dev/eval split in 4:1 ratio | 88 dev and 22 eval images in [manifest](data/manifest.json); 21 whole scene groups; [leakage audit](outputs/leakage_audit.json) |
| Task 2: classical segmentation | [Image-only classical implementation](src/classical.py); 110 binary [prediction masks](outputs/predictions) |
| Task 2: experiment log including unsuccessful trials | Ten executed trials in [experiments.json](outputs/experiments.json); report documents changes, motivations, parameters, development results and decisions |
| Task 3: metrics against pretrained references | [Per-image CSV](outputs/per_image_metrics.csv), [summary](outputs/summary.json), IoU, Dice, precision, recall, boundary F1 and scene-cluster bootstrap |
| Task 3: quantitative and qualitative analysis | [PDF report](report.pdf), [plots and failure figures](outputs/figures), four lowest-IoU eval samples and a successful sample |
| Task 4: process supplied YouTube clip and track eggs separately | Full 60-second, 1,440-frame [MP4](outputs/egg_tracking.mp4), two emitted IDs, [per-frame tracks](outputs/tracks.csv), [tracking implementation](src/track.py) |
| Public GitHub repository including PDF | This repository and [report.pdf](report.pdf) |
| PDF links to publicly accessible Drive tracking video | Clickable link on report page 1; `anyone/reader` permission verified |
| Setup and experiment instructions | [README.md](README.md), pinned [requirements.txt](requirements.txt), [recorded environment](outputs/environment.json) |
| Brief disclosure of AI assistance | README and report disclose implementation, prompt specification, experimentation and writing assistance |
| Reviewable validation | Four passing [tests](tests/test_pipeline.py); [artifact validation](outputs/validation.json) checks all masks, metric rows and every decoded video frame |

## Limits that remain part of the result

The assignment asks the classical method to try to match the pretrained method. It does not require a minimum metric score. This implementation achieves evaluation mean IoU 0.515 and Dice 0.617 and documents its failures. Reference masks are model-generated and uncertain in some submerged-egg scenes. An early exploratory run exposed evaluation results, so the final scene-disjoint evaluation is not a pristine blind test. Two emitted track IDs do not prove zero identity switches. These limits are disclosed in the report.

## Applicant actions

The assignment requires submission within seven days of receipt. The date of receipt is not established in this workspace, so deadline compliance cannot be verified here. Submit the public repository link through the application conversation, and study the implementation before the interview. No message has been sent to the recruiter on your behalf.
