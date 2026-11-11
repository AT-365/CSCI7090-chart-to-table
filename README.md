# Chart-to-Table Extraction for Vertical Bar Charts

**CSCI 7090: Advanced Computer Vision and Deep Learning, Final Project**
Autenia Murray, Georgia Southern University, Fall 2026

This project reads a picture of a vertical bar chart and writes out the data table behind it
(one `label | value` row per bar). It uses the Kaggle Benetech "Making Graphs Accessible" data.

- **Phase A (midterm):** compared DePlot and MatCha on a pilot set. MatCha won.
- **Phase B (final):** answered the midterm feedback, fine-tuned MatCha on a 3,500-chart subset
  with early stopping, built an OpenCV + Tesseract OCR baseline, and ran one final evaluation on a
  held-out test set (real and generated charts reported separately).

## Final test results (525 held-out charts, strict Benetech score)

| Test charts | MatCha (fine-tuned) | OpenCV + OCR baseline | Difference (95% bootstrap interval) |
|---|---|---|---|
| Real (69) | 0.936 | 0.349 | +0.587 (+0.481 to +0.693) |
| Generated (456) | 0.969 | 0.550 | +0.420 (+0.380 to +0.460) |
| All (525) | 0.965 | 0.523 | +0.442 (+0.405 to +0.480) |

The slides walk through every step, including the error analysis.

## What is in this repository

| Path | What it is |
|---|---|
| `V3_4_7_CSCI7090_Colab_PhaseB.ipynb` | The main notebook (Phase B), with saved outputs. Steps 1 to 13 run top to bottom. |
| `V3_4_6-Code/chart_eval.py` | Table parsing and Benetech-style scoring |
| `V3_4_6-Code/t4_pipeline.py` | Environment checks, generation, matched trainer, scoring helpers |
| `V3_4_6-Code/_v332_metric_original.py` | Reference copy of the original metric, used by the tests |
| `V3_4_6-Code/test_*.py` | Unit tests (standard-library `unittest`) |
| `V3_4_6-Code/SHA256SUMS.txt` | Checksums the notebook verifies in cell 1c |
| `slides/` | Final presentation (PowerPoint, with speaker notes) |

The modules in `V3_4_6-Code/` are the midterm (V3.4.6) code, reused unchanged in Phase B.

## How to run it

1. **Data.** Download the competition data from Kaggle
   (<https://www.kaggle.com/competitions/benetech-making-graphs-accessible/data>).
   It is not included here because of its size and license.
2. **Code.** Zip the files in `V3_4_6-Code/` into `V3_4_6-Code.zip`. Put that zip, the Kaggle zip
   and the notebook in one Google Drive folder. Cell 1a sets the folder paths; edit them to match.
3. **Runtime.** Google Colab (I ran it through the VS Code Colab extension). Steps 1 to 3 run on CPU.
   Training and generation (Step 4 onward) need a T4 GPU.
4. Run the cells in order. Expensive cells save their results to Drive, so a disconnect can resume.

To run the unit tests locally:

```bash
cd V3_4_6-Code
python -m unittest test_chart_eval
```

## Not included

Model checkpoints, the dataset and intermediate prediction files are left out because of size.
The notebook's saved outputs show the results of each step.
