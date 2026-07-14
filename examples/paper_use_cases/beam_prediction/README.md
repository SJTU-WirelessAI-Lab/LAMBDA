# RGB-only 60 GHz beam prediction

This package trains an RGB beam classifier on the LAMBDA Open Ground source
scene, then evaluates zero-shot or few-shot transfer on DeepSense Scenario 23.

```bash
python -m beam_prediction --help
python -m beam_prediction \
  --data-root /path/to/prepared/LAMBDA \
  --deepsense-root /path/to/scenario23_dev \
  --stride 200 --limit_train_per_scene 128 \
  --limit_test 256 --epochs 1 --batch_size 16 --num_workers 2 \
  --no_pretrained
```

`LAMBDA_DATA_ROOT` and `DEEPSENSE_ROOT` can be set instead of passing the two
root arguments. `DEEPSENSE_ROOT` must contain `scenario23.csv`, with
`unit1_rgb` and `unit1_pwr_60ghz` paths relative to that root, plus the
`unit1_beam_index` label column. See the parent README for the complete
expected layout. DeepSense data is external and is not distributed here.
