# USPTO-31K Full Unmatched Fine-Tuning

Use `finetune_uspto31k_full_unmatched.py` with the JSON config to fine-tune the best checkpoint on your own USPTO-31K split files.

Example:

```bash
python fine_tuning/finetune_uspto31k_full_unmatched.py --config fine_tuning/config_uspto31k_full_unmatched.json
```

Useful knobs in the JSON config:

- `epochs`
- `batch_size`
- `lr_encoder`
- `lr_decoder`
- `patience`
- `beam_eval_examples`
- `valid_eval_examples`
- `test_eval_examples`
- `random_eval_subset`
- `load_optimizer_state`
- `evaluate_test_each_epoch`

Provide `train_path`, `valid_path`, `test_path`, `train_target_path`, `valid_target_path`, and `test_target_path` in the JSON config, or place `src-train.txt`, `src-val.txt`, `src-test.txt`, `tgt-train.txt`, `tgt-val.txt`, and `tgt-test.txt` in the data directory.

The script writes outputs to `output/finetuning_uspto31k_full_unmatched/` by default:

- `training_log.csv`
- `best_checkpoint.pt`
- `last_checkpoint.pt`
- `dataset_report.json`
- `final_test_report.json`
