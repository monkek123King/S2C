python train.py --experiment_name piano2posi_LR --bs_dim 6 --adjust --is_random --up_list 1467634 66685747 \
--data_root ./ --iterations 200000 --batch_size 8 --train_sec 8 --feature_dim 512 \
--wav2vec_path ./checkpoints/hubert-large-ls960-ft --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
--latest_layer tanh --encoder_type transformer --num_layer 4