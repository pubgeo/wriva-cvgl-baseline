## Full workflow for building the test-only dataset from WRIVA-CVGL
python create_splits.py \
  --root_dir /home/yejz1/wriva/data/WRIVA-CVGL-Datasets-V2 \
  --out_dir splits/wriva_release \
  --train_ratio 0 \
  --val_ratio 0 \
  --test_ratio 1 \
  --seed 42 \
  --output_relative_paths \
  --validate_with_reference

python build_image_pair_manifest.py \
  --pairs-txt splits/wriva_release/test_split.txt \
  --dataset-root /home/yejz1/wriva/data/WRIVA-CVGL-Datasets-V2  \
  --output splits/wriva_release/test_manifest.pkl \
  --no-validate-paths


## Full workflow for building the train/val/test dataset from Visym-CVGL
# python create_splits.py \
#   --root_dir /home/yejz1/wriva/data/visym-cvgl-final \
#   --out_dir splits/visym801010 \
#   --train_ratio 0.8 \
#   --val_ratio 0.1 \
#   --test_ratio 0.1 \
#   --seed 42 \
#   --output_relative_paths \
#   --validate_with_reference

# python build_image_pair_manifest.py \
#   --pairs-txt splits/visym801010/train_split.txt \
#   --dataset-root /home/yejz1/wriva/data/visym-cvgl-final  \
#   --output splits/visym801010/train_manifest.pkl \
#   --no-validate-paths

# python build_image_pair_manifest.py \
#   --pairs-txt splits/visym801010/train_split.txt \
#   --dataset-root /home/yejz1/wriva/data/visym-cvgl-final  \
#   --output splits/visym801010/train_manifest.pkl \
#   --no-validate-paths

# python build_image_pair_manifest.py \
#   --pairs-txt splits/visym801010/train_split.txt \
#   --dataset-root /home/yejz1/wriva/data/visym-cvgl-final  \
#   --output splits/visym801010/train_manifest.pkl \
#   --no-validate-paths