python train_diffusion.py --experiment_name piano2pose --is_random --unet_dim 256 --iterations 800000 \
--bs_dim 96  --batch_size 16 --train_sec 8 --data_root ./ \
--xyz_guide  --check_val_every_n_iteration 1000 --save_every_n_iteration 1000 \
--adjust --piano2posi_path logs/piano2posi_LR --encoder_type transformer --num_layer 4 \
--lr 1e-5 --fusion 4 --obj pred_v