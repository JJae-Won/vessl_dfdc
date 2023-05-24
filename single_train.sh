git clone https://github.com/NVIDIA/apex
sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  ./apex

python training/pipelines/train_classifier_vessl.py \
    --config configs/b7.json \
     --freeze-epochs 0 --test_every 1 --opt-level O1 \
     --label-smoothing 0.01 --folds-csv all_folds.csv \
      --fold 1 --seed 111 --data-dir /input --prefix b7_111_ > logs/b7_111