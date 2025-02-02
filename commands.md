## Fine-Tune model

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

## Train n-gram models

python NeMo/scripts/asr_language_modeling/ngram_lm/train_kenlm.py \
 nemo_model_file="stt_fa_fastconformer_hybrid_large_finetuned.nemo" \
 train_paths=["combined_corpus.txt"] \
 kenlm_bin_path="kenlm/build/bin/" \
 kenlm_model_file="../3_gram_model" \
 ngram_length=3 \

## Evaluate model

### Evaluation of fine-tuned model alone

python NeMo/examples/asr/speech_to_text_eval.py \
 model_path="./stt_fa_fastconformer_hybrid_large_finetuned.nemo" \
 dataset_manifest="test_manifest.json" \
 batch_size=8 \
 use_cer=True

### Evaluation of fine-tuned model + n-gram language model

python eval_beamsearch_ngram_transducer.py nemo_model_file="./stt_fa_fastconformer_hybrid_large_finetuned.nemo" \
 input_manifest="test_manifest.json" \
 kenlm_model_file="4_gram_model" \
 probs_cache_file=null \
 decoding_mode=beamsearch_ngram \
 decoding_strategy="maes"
