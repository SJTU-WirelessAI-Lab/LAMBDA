# Cross-dataset RGB beam prediction

This package provides a cross-dataset beam-prediction workflow:

- source: LAMBDA Open Ground (`square_3ss` in the original experiment tree);
- target: DeepSense Scenario 23;
- input: RGB images;
- classifier: ImageNet-initialized ResNet-50;
- output: 64 classes in the sky-up beam codebook;
- target labels: `unit1_beam_index` from `scenario23.csv`.

```bash
python -m beam_prediction \
  --data-root /path/to/prepared/LAMBDA \
  --deepsense-root /path/to/scenario23 \
  --do_few_shot
```

The default few-shot evaluation uses 64, 128, 256, 512, and 1024 target samples
and reinitializes the 64-way classification head before each adaptation run.

`LAMBDA_DATA_ROOT` and `DEEPSENSE_ROOT` can be set instead of passing the two
root arguments. `DEEPSENSE_ROOT` must contain `scenario23.csv` and the Unit 1
RGB files referenced by its `unit1_rgb` column. DeepSense data is external and
is not distributed here; see the parent README for data preparation and
citation guidance.
