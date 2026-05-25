# PhysLLM

Public release of the PhysLLM model code and its core training/inference path.

## Scope

This repository contains:

- the PhysLLM and NewPhysLLM model code
- the core data loaders used for `UBFC-rPPG`, `PURE`, `BUAA`, and `MMPD`
- evaluation code
- example configs for multi-source training and held-out testing

This repository does not contain:

- private datasets or preprocessed caches
- experiment logs and result files
- model weights or third-party checkpoints
- local environment folders

## Checkpoints

The public release expects external checkpoints to be passed in through config fields:

- `MODEL.CHECKPOINTS.VIDEO_ENCODER`
- `MODEL.CHECKPOINTS.FACE_ENCODER`
- `INFERENCE.MODEL_PATH`

You can also provide encoder checkpoints through environment variables:

- `PHYSLLM_VIDEO_ENCODER_CKPT`
- `PHYSLLM_FACE_ENCODER_CKPT`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example

Train:

```bash
python main.py --config_file configs/physllm_multisource_train_example.yaml
```

Test:

```bash
python main.py --config_file configs/physllm_only_test_zpu_example.yaml
```

## Notes

- The multi-source example keeps `ZPU/ZPH` as a held-out test domain.
- The example configs use placeholder paths and are meant to be edited locally.
- The repository includes only the PhysLLM-related code path, not the full original research workspace.
