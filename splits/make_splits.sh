python create_splits.py \
  --root_dir /home/yejz1/wriva/data/visym-cvgl-final \
  --out_dir splits/visym \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --seed 42 \
  --output_relative_paths \
  --validate_with_reference