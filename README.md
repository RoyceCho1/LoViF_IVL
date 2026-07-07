# FUMO Final Inference

`main_inference.py` is the single entry point for the packaged inference flow.

```text
data/test/LQ -> data/test/IS -> data/test/P_INT -> data/test/IS2 -> make_mask -> stage3 -> result
```

## Layout

```text
final/
  data/test/
    LQ/                # input images
    IS/                # stage1 output
    P_INT/             # Qwen3-VL priors
    IS2/               # stage2 output
    drop_mask/         # make_mask raindrop masks
    reflection_mask/   # make_mask reflection masks
    stage3_runs/       # optional stage3 run artifacts
  stage1/
  stage2/
  stage3/
  weights/
    stage1/
    stage2/
    stage3/
  result/              # final output PNGs only
  main_inference.py
  README.md
  requirements.txt
```

## Environment

```bash
cd /home/student_1/LoViF/FUMO/final
conda create -n lovif python=3.10 -y
conda activate lovif
pip install -r requirements.txt
```

If Qwen3-VL import fails in `transformers`:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

Check Qwen3-VL support before running P_INT:

```bash
python - <<'PY'
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import qwen_vl_utils
print("Qwen3-VL import OK")
PY
```

If Hugging Face model download requires authentication:

```bash
hf auth login
hf auth whoami
```

Optional pre-download:

```bash
hf download Qwen/Qwen3-VL-8B-Instruct
```

## Weights

Weights are distributed separately from `final.zip` because of file size. After extracting `final.zip`, download the weights and place them under `final/weights/` as follows.

```text
final/weights/
  stage1/
    best_ckpt
  stage2/
    controlnet/
    unet/
    nafnet_refine.pth
    nafnet_refine_head.pth
  stage3/
    latest.pth
```

Google Drive links for the weight package can be added here later.

## Default Execution Mode

`main_inference.py` is configured to use one visible GPU by default. CPU worker counts are set in `SETTINGS`.

```text
CUDA_VISIBLE_DEVICES=0
P_INT max_workers=1
stage2 num_shards=1
make_mask num_workers=16
stage3 num_workers=16
```

## Run

Put input images in:

```text
data/test/LQ/
```

Edit `SETTINGS` in `main_inference.py`, then run:

```bash
python main_inference.py
```

Set `run_stages` in `main_inference.py` for the available weights. The final pipeline uses:

```python
"run_stages": ["stage1", "p_int", "stage2", "make_mask", "stage3"]
```

`make_mask` creates `drop_mask` from `LQ + IS` and `reflection_mask` from `IS + IS2`. `stage3` then loads the stage3 weight and blends the final result.

Check the planned flow without running models:

```bash
python main_inference.py --dry-run
```

Run only one stage:

```bash
python main_inference.py --stage stage1
python main_inference.py --stage p_int
python main_inference.py --stage stage2
python main_inference.py --stage make_mask
python main_inference.py --stage stage3
```

## File Matching

Filenames are preserved between stages:

```text
data/test/LQ/example.png
data/test/IS/example.png
data/test/P_INT/example.npy
data/test/IS2/example.png
data/test/drop_mask/example.png
data/test/reflection_mask/example.png
result/example.png
```

All stages match files by the same filename stem. For example, `example.png` uses `example.npy` as its P_INT prior.
