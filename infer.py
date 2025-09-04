import argparse
import copy
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from scipy.signal import savgol_filter

from datasets.PianoPose import PianoPose
from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D
from datasets.show import render_result
from models.evaluate import fid, loc_distance, MW2, acceleration, EmbeddingSpaceEvaluator, acceleration_2

import time


DEBUG = 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_path", type=str, default="logs/diffusion_posiguide")
    # dataset
    parser.add_argument("--data_root", type=str, default='/hdd/data3/piano-bilibili')
    parser.add_argument("--valid_batch_size", type=int, default=1)
    parser.add_argument("--mode", type=str, default='test')
    args = parser.parse_args()
    return args

def main():
    # get args
    args_exp = get_args()
    args = copy.copy(args_exp)
    with open(args.exp_path + '/args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    args.preload = False
    args.data_root = args_exp.data_root
    args.up_list = ['1467634', '66685747']

    # get dataloader
    args.is_random = False
    valid_dataset = PianoPose(args=args, phase=args_exp.mode)

    args_piano2posi = copy.copy(args)
    with open(args_exp.exp_path + '/args_posi.txt', 'r') as f:
        dic = json.load(f)
        dic['hidden_type'] = args.hidden_type if 'hidden_type' in args.__dict__.keys() else 'audio_f'
        args_piano2posi.__dict__ = dic
    piano2posi_left = Piano2Posi(args_piano2posi)
    piano2posi_right = copy.deepcopy(piano2posi_left)
    if 'hidden_type' in args.__dict__.keys():
        if args.hidden_type == 'audio_f':
            cond_dim = 768 if 'base' in args_piano2posi.wav2vec_path else 1024
        elif args.hidden_type == 'hidden_f':
            cond_dim = args_piano2posi.feature_dim
        elif args.hidden_type == 'both':
            cond_dim = args_piano2posi.feature_dim + (768 if 'base' in args_piano2posi.wav2vec_path else 1024)
    else:
        cond_dim = 768 if 'base' in args_piano2posi.wav2vec_path else 1024

    unet = Unet1D(
        dim=args.unet_dim,
        dim_mults=(1, 2, 4, 8),
        channels=args.bs_dim // 2,
        remap_noise=args.remap_noise if 'remap_noise' in args.__dict__ else True,
        condition=True,
        guide=args.xyz_guide,
        guide_dim=6 if args.xyz_guide else 0,
        condition_dim=cond_dim,
        encoder_type=args.encoder_type if 'encoder_type' in args.__dict__ else 'none',
        num_layer=args.num_layer if 'num_layer' in args.__dict__ else None,
        feature_fusion=args.fusion,
        # feature_fusion=3,
    )

    timesteps = args.timesteps

    model_left = GaussianDiffusion1D_piano2pose(
        unet,
        piano2posi_left,
        seq_length=args.train_sec * 30,
        timesteps=timesteps,
        objective='pred_v',
    )

    model_right = GaussianDiffusion1D_piano2pose(
        copy.deepcopy(unet),
        piano2posi_right,
        seq_length=args.train_sec * 30,
        timesteps=timesteps,
        objective='pred_v',
    )

    model_path_left = sorted([i for i in os.listdir(args_exp.exp_path) if 'ckpt' in i and 'left' in i and '609000' in i], key=lambda x: int(x.split('-')[2].split('=')[1]))[-1]
    model_path_right = sorted([i for i in os.listdir(args_exp.exp_path) if 'ckpt' in i and 'right' in i and '609000' in i], key=lambda x: int(x.split('-')[2].split('=')[1]))[-1]
    print('left hand load from:', model_path_left)
    print('right hand load from:', model_path_right)
    model_left.load_state_dict(torch.load(args_exp.exp_path + '/' + model_path_left, map_location='cpu')['state_dict'])
    model_right.load_state_dict(torch.load(args_exp.exp_path + '/' + model_path_right, map_location='cpu')['state_dict'])
    model_left.to('cuda')
    model_right.to('cuda')


    scale = torch.tensor([1.5, 1.5, 25]).cuda()
    test_ids = np.arange(0, 1) # PLEASE MODIFY TO WITCH IDS TO INFER
    os.makedirs(f'results/{args.experiment_name}', exist_ok=True)
    with torch.no_grad():
        # for test_id in range(len(valid_dataset)):
        for test_id in test_ids:
            batch, para = valid_dataset.__getitem__(test_id, True)
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key]).cuda().unsqueeze(0)

            audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']
            frame_num = left_pose.shape[1]

            gt_left = left_pose[:, :, 4:].permute(0, 2, 1) / np.pi          # B, 48, N
            gt_right = right_pose[:, :, 4:].permute(0, 2, 1) / np.pi        # B, 48, N

            begin_time = time.time()

            guide_left, cond_left = model_left.getGuideAndCond(gt_left, audio=audio)
            guide_right, cond_right = model_right.getGuideAndCond(gt_right, audio=audio)

            guide_left_total = torch.cat([guide_left, guide_right], 2)
            guide_right_total = torch.cat([guide_right, guide_left], 2)

            pose_hat_left, pose_hat_right = model_left.sample(guide_left_total, cond_left, guide_right_total, cond_right, model_right.model, gt_left.shape[0])

            end_time = time.time()
            print(f'Test {test_id} done, time: {end_time - begin_time:.2f}s')

            pose_hat = torch.cat([pose_hat_right, pose_hat_left], 1)
            pose_hat = pose_hat.permute(0, 2, 1)
            guide = torch.cat([guide_right, guide_left], 2)

            right_pose_pred = (pose_hat[:, :, :args.bs_dim // 2] * np.pi).cpu().numpy()
            left_pose_pred = (pose_hat[:, :, args.bs_dim // 2:] * np.pi).cpu().numpy()
            right_trans_pred = (guide[:, :, :3] * scale).cpu().numpy()
            left_trans_pred = (guide[:, :, 3:] * scale).cpu().numpy()
            right_trans = right_pose[:, :, 1:4].cpu().numpy()
            left_trans = left_pose[:, :, 1:4].cpu().numpy()
            right_poses = right_pose[:, :, 4:].cpu().numpy()
            left_poses = left_pose[:, :, 4:].cpu().numpy()

            # guide = guide.permute(0, 2, 1)
            prediction = pose_hat[0].detach().cpu().numpy() * np.pi
            guide = (guide * scale.repeat(2))[0].cpu().numpy()

            for i in range(prediction.shape[1]):
                prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
            for i in range(guide.shape[1]):
                guide[:, i] = savgol_filter(guide[:, i], 5, 2)

            os.makedirs(f'results/{args.experiment_name}/pred_{test_id}', exist_ok=True)
            os.makedirs(f'results/{args.experiment_name}/pred_{test_id}/pic', exist_ok=True)
            os.makedirs(f'results/{args.experiment_name}/gt_{test_id}', exist_ok=True)
            os.makedirs(f'results/{args.experiment_name}/gt_{test_id}/pic', exist_ok=True)

            right_fid = fid(right_poses[0], right_pose_pred[0])
            left_fid = fid(left_poses[0], left_pose_pred[0])
            right_loc_dist = loc_distance(right_trans[0, :, :2], right_trans_pred[0, :, :2])
            left_loc_dist = loc_distance(left_trans[0, :, :2], left_trans_pred[0, :, :2])

            sum = right_fid + left_fid + right_loc_dist + left_loc_dist

            with open(f'results/{args.experiment_name}/pred_{test_id}/metrics.txt', 'w') as f:
                f.write(f'fid_right: {right_fid:.4f}\n')
                f.write(f'fid_left: {left_fid:.4f}\n')
                f.write(f'loc_dist_right: {right_loc_dist:.4f}\n')
                f.write(f'loc_dist_left: {left_loc_dist:.4f}\n')
                f.write(f'sum: {sum:.4f}\n')
            
            render_result(f'results/{args.experiment_name}/pred_{test_id}/pic',
                          audio[0].cpu().numpy(),
                          np.concatenate([guide[:, :3], prediction[:, :48]], 1),
                          np.concatenate([guide[:, 3:], prediction[:, 48:]], 1), False, [120, 150, 255]) 
            render_result(f'results/{args.experiment_name}/gt_{test_id}/pic',
                          audio[0].cpu().numpy(),
                          right_pose[0, :, 1:52].cpu().numpy(),
                          left_pose[0, :, 1:52].cpu().numpy(), False, [180, 150, 120])


if __name__ == "__main__":
    main()
