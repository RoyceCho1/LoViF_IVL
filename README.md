# FUMO Final Inference

`main_inference.py` runs the full inference pipeline:

```text
LQ -> stage1(IS) -> P_INT -> stage2(IS2) -> make_mask -> stage3 -> result
```

## Setup

```bash
conda create -n lovif python=3.10 -y
conda activate lovif

# Install PyTorch first. The default command is for the CUDA wheel used on our RTX 5090 setup.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the remaining packages.
pip install -r requirements.txt
```

Check that PyTorch can use CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

If `cuda available` is `False`, reinstall `torch` and `torchvision` with the CUDA wheel that matches your machine from https://pytorch.org/get-started/locally/.

Check Qwen3-VL support:

```bash
python - <<'PY'
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import qwen_vl_utils
print("Qwen3-VL import OK")
PY
```

If the import fails:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

If Hugging Face authentication is needed:

```bash
hf auth login
```

## Weights

Weights are provided separately because of file size. The expected structure is the same as the `weights/` directory in this package.

Weights are available here: [Google Drive](https://drive.google.com/drive/folders/19a4FNc54ase259l22RUy9o4IPlF-3xYa?usp=drive_link)

If you receive `weights.zip`, extract it at the project root so that the paths below exist directly under `weights/`:

```text
weights/
  stage1/
    best_ckpt
  stage2/
    controlnet/
      config.json
      diffusion_pytorch_model.safetensors
    unet/
      config.json
      diffusion_pytorch_model.safetensors
    nafnet_refine.pth
    nafnet_refine_head.pth
  stage3/
    best.pth
```

You can also place the files manually as long as the same paths and filenames are preserved. These paths are used by `main_inference.py`.

## Input

Put input images in:

```text
data/test/LQ/
```

The pipeline writes intermediate files to:

```text
data/test/IS/
data/test/P_INT/
data/test/IS2/
data/test/drop_mask/
data/test/reflection_mask/
```

Final output PNGs are written to:

```text
result/
```

## Runtime Note

`P_INT` generation uses Qwen3-VL, so its runtime can vary significantly depending on the GPU. On our RTX 5090 setup, `P_INT` generation takes about 2.8 seconds per image.

## Overwrite / Resume

`SETTINGS["skip_existing"]` in `main_inference.py` controls whether existing generated files are reused or regenerated.

```python
"skip_existing": False  # overwrite existing generated outputs
"skip_existing": True   # reuse existing outputs when possible
```

This setting is shared by stage1, P_INT, stage2, and mask generation. Use `False` when you want to rerun the full pipeline from the current inputs and settings.

## Run

Check paths and stages without running models:

```bash
python main_inference.py --dry-run
```

Run the full pipeline:

```bash
python main_inference.py
```

Run a single stage:

```bash
python main_inference.py --stage stage1
python main_inference.py --stage p_int
python main_inference.py --stage stage2
python main_inference.py --stage make_mask
python main_inference.py --stage stage3
```

## File Matching

Files are matched by filename stem:

```text
data/test/LQ/example.png
data/test/IS/example.png
data/test/P_INT/example.npy
data/test/IS2/example.png
data/test/drop_mask/example.png
data/test/reflection_mask/example.png
result/example.png
```
