python training/pipelines/train_classifier_vessl.py \
    --config configs/b7.json \
     --freeze-epochs 0 --test_every 1 --opt-level O1 \
     --label-smoothing 0.01 --folds-csv all_folds.csv \
      --fold 1 --seed 111 --data-dir /input --prefix b7_111_ > logs/b7_111