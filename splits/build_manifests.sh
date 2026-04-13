SPLIT_TYPE="train"

python build_image_pair_manifest.py \
  --pairs-txt splits/visym/${SPLIT_TYPE}_split.txt \
  --dataset-root /home/yejz1/wriva/data/visym-cvgl-final  \
  --output splits/visym/${SPLIT_TYPE}_manifest.pkl \
  --no-validate-paths