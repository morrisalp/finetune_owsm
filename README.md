# Wav2Gloss: Generating Interlinear Glossed Text from Speech
- Accepted to ACL 2024
- Source code for reproducing E2E OWSM-finetuned model
- Dataset: https://huggingface.co/datasets/wav2gloss/fieldwork
- Paper: https://arxiv.org/abs/2403.13169
- XLS-R/WavLM baseline: https://github.com/juice500ml/espnet/tree/wav2gloss/
- Poster: [poster.pdf](./poster.pdf)
- Slides: [slides.pdf](./slides.pdf)
- 10-minute presentation: https://drive.google.com/file/d/1nSelG6a56924JYMnnvKysb805R-lSZIa/view?usp=sharing

## Prerequisites

### Obtaining the Dataset

Use `download_data.sh` to automaticall download and extract the [Fieldwork dataset](https://huggingface.co/datasets/wav2gloss/fieldwork). The script requires [git lfs](https://git-lfs.com/).

```sh
./download_data.sh
```

### Installation
```sh
conda create -p ./envs
conda activate ./envs

conda install python=3.10 pytorch=2.0 pytorch-cuda=11.8 torchaudio -c pytorch -c nvidia
pip install -r requirements.txt
```

## Model Training
```sh
# Supported tasks
"transcription", "underlying", "gloss", "translation"

# Supported languages
"taba1259", "tond1251", "kach1280", "arta1239", "vera1241",
"sanz1248", "sumb1241", "nort2641", "kara1499", "mand1415",
"tehr1242", "taul1251", "ainu1240", "even1259", "dolg1241",
"kama1378", "selk1253", "komn1238", "sout2856", "apah1238",
"teop1238", "jeju1234", "ruul1235", "sumi1235", "beja1238",
"kaka1265", "goro1270", "savo1255", "texi1237", "pnar1238",
"nngg1234", "arap1274", "port1286", "trin1278", "bora1263",
"slav1254", "balk1252"

# Single-task Monolingual training
# Example: task=transcription, language=dolg1241
python finetune.py \
    --tasks transcription \
    --langs dolg1241 \
    --devices 1 \
    --batch_size 16 \
    --max_epochs 30 \
    --exp_name transcription_dolg1241

# Single-task Multilingual training
# Example: task=transcription
python finetune.py \
    --tasks transcription \
    --devices 1 \
    --batch_size 16 \
    --exp_name transcription_full

# Multi-task Multilingual training
python finetune.py \
    --devices 1 \
    --batch_size 16 \
    --exp_name all_full

# You can try different pre-trained models by feeding parameters such as
--model_name espnet/owsm_v3.1_ebf
--model_name espnet/owsm_v3.2
```

## Model evaluation
```sh
# Example PATH_TO_CKPT
PATH_TO_CKPT=exps/translation_full_2024-02-06T13:50:02.501140/lightning_logs/version_141479/checkpoints/epoch=5-step=16548.ckpt

python evaluate_model.py \
    --devices 1 \
    --checkpoint_path $PATH_TO_CKPT
```

## CHANGELOG: Recent Updates

* Update installation with missing dependencies and avoid script that corrupts current conda env
* Add flag for dataloader workers (`--num_workers`) and lower default to avoid errors with too many open files