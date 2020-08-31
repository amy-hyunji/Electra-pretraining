CUDA_VISIBLE_DEVICES=7 python3 run_finetuning.py --data-dir ./dataset/ --model-name electra_small_wordnet --hparams '{"model_size": "small", "task_names": ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst", "sts"]}'

