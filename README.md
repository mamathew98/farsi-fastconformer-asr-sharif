# Farsi FastConformer ASR

A comprehensive Automatic Speech Recognition (ASR) project for Persian (Farsi), leveraging NVIDIA's FastConformer architecture. This repository covers dataset preprocessing, fine-tuning the `nvidia/stt_fa_fastconformer_hybrid_large` model, training n-gram language models, and evaluation scripts. Also included is a Gradio demo application for quick testing and deployment.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset Description](#dataset-description)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Model Fine-Tuning](#model-fine-tuning)
6. [Language Model Training](#language-model-training)
7. [Usage](#usage)
8. [Gradio Demo](#gradio-demo)
9. [License](#license)
10. [Contact](#contact)

---

## Introduction
This project focuses on building and evaluating a Persian (Farsi) ASR system using the FastConformer architecture from NVIDIA. We collected and processed a large YouTube-based dataset (`ASR Farsi YouTube Chunked 10 Seconds`) comprising diverse speech, dialects, and noisy conditions to ensure robust real-world performance.

---

## Features
- **Data Preprocessing**: Scripts for trimming silence, noise reduction, and text normalization.
- **FastConformer Fine-Tuning**: Training scripts and configs for the `nvidia/stt_fa_fastconformer_hybrid_large` model.
- **Language Model Training**: N-gram LM (KenLM) training for Bi-gram, Tri-gram, and 4-gram.
- **Evaluation & Metrics**: Scripts for WER/CER calculation and beam search decoding with LM.
- **Gradio App**: A user-friendly demo interface to test the model in real-time.

---

## Dataset Description
- **Name**: ASR Farsi YouTube Chunked 10 Seconds
- **Duration**: ~1200 hours of diverse Persian speech (15k+ speakers, 18 dialects)
- **Sampling Rate**: 16 kHz, 16-bit PCM
- **Source**: Public YouTube videos with Creative Commons
- **Key Challenges**: Real-world noise, code-switching, dialect variability

---

## Preprocessing Pipeline
We employ a multi-stage pipeline for data cleaning and normalization:
1. **Silence Removal** with `librosa.effects.trim`.
2. **Noise Reduction** via `noisereduce`.
3. **Text Normalization** for Arabic-to-Persian character replacements, diacritic removal.
4. **Metadata Management** using JSON manifests.

   ---

## Model Fine-Tuning
We use NVIDIA’s pretrained [`stt_fa_fastconformer_hybrid_large.nemo`](https://huggingface.co/nvidia) model and fine-tune it on approximately 40% of the dataset to reduce computational cost. Key training parameters:
- **Optimizer**: AdamW
- **Epochs**: 10
- **Batch Size**: 8
- **Learning Rate**: 1e-4

  **Commands**:
```bash
# Example fine-tuning command
python NeMo/examples/asr/speech_to_text_finetune.py ...
```
---

# Language Model Training
We train Bi-gram, Tri-gram, and 4-gram models with KenLM on the combined corpus (combined_corpus.txt). These models are then used in beam search decoding for improved performance on real-world data.

```bash
python NeMo/scripts/asr_language_modeling/ngram_lm/train_kenlm.py ...
```

---

##Evaluation
Evaluation is performed both on the fine-tuned model alone and with different n-gram LMs. Metrics:

WER (Word Error Rate)
CER (Character Error Rate)

Results show a significant drop in WER from ~68% (original model) to ~32% (fine-tuned + 4-gram LM)

---

## Usage

1- Clone the repo:
```bash
git clone https://github.com/your-username/farsi-fastconformer-asr.git
```

2- Install requirements:
```bash
pip install -r requirements.txt
```

3- Preprocess data
Use the notebokk

4- Fine-tune the model
```bash
python NeMo/examples/asr/speech_to_text_finetune.py \
 --config-path="./conf/asr_finetune" \
 --config-name="speech_to_text_finetune" \
 init_from_nemo_model="stt_fa_fastconformer_hybrid_large.nemo" \
 model.tokenizer.dir="./tokenizer/tokenizer_spe_bpe_v1024" \
 model.train_ds.manifest_filepath="train_manifest2.json" \
 model.train_ds.batch_size=8 \
 model.validation_ds.batch_size=8 \
 model.validation_ds.manifest_filepath="val_manifest.json" \
 exp_manager.exp_dir="Stt_fa_fastconformer_hybrid_large_finetuned" \
 trainer.devices=-1 \
 trainer.max_epochs=10 \
 model.optim.name="adamw" \
 model.optim.lr=1e-4 \
 model.optim.betas=[0.9,0.999] \
 model.optim.weight_decay=0.0001 \
 model.optim.sched.warmup_steps=2000
```

5- Train n-gram models
```bash
python NeMo/scripts/asr_language_modeling/ngram_lm/train_kenlm.py \
 nemo_model_file="stt_fa_fastconformer_hybrid_large_finetuned.nemo" \
 train_paths=["combined_corpus.txt"] \
 kenlm_bin_path="kenlm/build/bin/" \
 kenlm_model_file="../3_gram_model" \
 ngram_length=3 \
```

6- Evaluation of fine-tuned model alone
```bash
python NeMo/examples/asr/speech_to_text_eval.py \
 model_path="./stt_fa_fastconformer_hybrid_large_finetuned.nemo" \
 dataset_manifest="test_manifest.json" \
 batch_size=8 \
 use_cer=True
```

7- Evaluation of fine-tuned model + n-gram language model
```bash
python eval_beamsearch_ngram_transducer.py nemo_model_file="./stt_fa_fastconformer_hybrid_large_finetuned.nemo" \
 input_manifest="test_manifest.json" \
 kenlm_model_file="4_gram_model" \
 probs_cache_file=null \
 decoding_mode=beamsearch_ngram \
 decoding_strategy="maes"
```

---

## Gradio Demo
- https://huggingface.co/spaces/Pooya-Fallah/FastConformer_Persian
- https://huggingface.co/spaces/Pooya-Fallah/FastConformer_Persian_Streaming
  
---

## License
MIT License

Copyright (c) [2024] 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

---

## Contact
For inquiries or collaborations, feel free to contact 
- [Mohammad Naseri] at: [mohammad.na3ri@gmail.com]
- [Pooya Fallah] at: [-]
