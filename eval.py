import argparse
import copy
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.utils.data as data
from thop import profile

from datasets.PianoPose import PianoPose
from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D
from models.evaluate import fid, loc_distance, MW2, acceleration, EmbeddingSpaceEvaluator, acceleration_2

DEBUG = 0

weights = {
    'right': {'fid': 0.25, 'MW': 0.15, 'dist': 0.1, 'smooth': 0.15},
    'left':  {'fid': 0.25, 'MW': 0.15, 'dist': 0.1, 'smooth': 0.15},
    'FID2': 0.3,
    'F_dist': 0
}

norm_range = {
    'right': {'fid': (0, 1), 'MW': (0, 1), 'dist': (0, 0.1), 'smooth': (0, 1)},
    'left': {'fid': (0, 1), 'MW': (0, 1), 'dist': (0, 0.1), 'smooth': (0, 1)},
    'FID2': (0, 1000),
    'F_dist': (0, 2000)
}

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_path", type=str, default="logs/diffusion_posiguide")
    # dataset
    parser.add_argument("--data_root", type=str, default='/hdd/data3/piano-bilibili')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--valid_batch_size", type=int, default=1)
    args = parser.parse_args()
    return args

def main(metrics_sum, model_path_left, model_path_right):
    # get args
    args_exp = get_args()
    args = copy.copy(args_exp)

    with open(args.exp_path + '/args.txt', 'r') as f:
        args.__dict__ = json.load(f)

    args.preload = False
    args.data_root = args_exp.data_root
    args.up_list = ['1467634', '66685747']

    # get dataloader
    valid_loader = data.DataLoader(
        PianoPose(args=args, phase=args_exp.mode),
        batch_size=args_exp.valid_batch_size)


    # load model
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

    if model_path_left is None:
        model_path_left = sorted([i for i in os.listdir(args_exp.exp_path) if 'ckpt' in i and 'left' in i], key=lambda x: int(x.split('-')[2].split('=')[1]))[-1]
        model_path_right = sorted([i for i in os.listdir(args_exp.exp_path) if 'ckpt' in i and 'right' in i], key=lambda x: int(x.split('-')[2].split('=')[1]))[-1]
    print('left hand load from:', model_path_left)
    print('right hand load from:', model_path_right)
    model_left.load_state_dict(torch.load(args_exp.exp_path + '/' + model_path_left, map_location='cpu')['state_dict'])
    model_right.load_state_dict(torch.load(args_exp.exp_path + '/' + model_path_right, map_location='cpu')['state_dict'])
    model_left.to('cuda')
    model_right.to('cuda')

    fid2 = EmbeddingSpaceEvaluator(embed_net_path='checkpoints/gesture_autoencoder_checkpoint_best.bin')
    scale = torch.tensor([1.5, 1.5, 25]).cuda()
    metrics = {'right': {'fid': [], 'MW': [], 'dist':[], 'smooth': []},
            'left': {'fid': [], 'MW': [], 'dist':[], 'smooth': []}}
    results = {'right_pose': [], 'right_trans': [], 'left_pose': [], 'left_trans': []}
    gts = {'right_pose': [], 'right_trans': [], 'left_pose': [], 'left_trans': []}
    with torch.no_grad():
        for v_idx, batch in enumerate(valid_loader):
            for key in batch.keys():
                batch[key] = batch[key].cuda()

            audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']

            frame_num = left_pose.shape[1]

            gt_left = left_pose[:, :, 4:].permute(0, 2, 1) / np.pi          # B, 48, N
            gt_right = right_pose[:, :, 4:].permute(0, 2, 1) / np.pi        # B, 48, N

            guide_left, cond_left = model_left.getGuideAndCond(gt_left, audio=audio)
            guide_right, cond_right = model_right.getGuideAndCond(gt_right, audio=audio)

            guide_left_total = torch.cat([guide_left, guide_right], 2)
            guide_right_total = torch.cat([guide_right, guide_left], 2)

            pose_hat_left, pose_hat_right = model_left.sample(guide_left_total, cond_left, guide_right_total, cond_right, model_right.model, gt_left.shape[0])

            pose_hat = torch.cat([pose_hat_right, pose_hat_left], 1)
            pose_hat = pose_hat.permute(0, 2, 1)
            guide = torch.cat([guide_right, guide_left], 2)

            # re-normalize
            right_pose_pred = (pose_hat[:, :, :args.bs_dim // 2] * np.pi).cpu().numpy()
            left_pose_pred = (pose_hat[:, :, args.bs_dim // 2:] * np.pi).cpu().numpy()
            right_trans_pred = (guide[:, :, :3] * scale).cpu().numpy()
            left_trans_pred = (guide[:, :, 3:] * scale).cpu().numpy()

            right_trans = right_pose[:, :, 1:4].cpu().numpy()
            left_trans = left_pose[:, :, 1:4].cpu().numpy()
            right_pose = right_pose[:, :, 4:].cpu().numpy()
            left_pose = left_pose[:, :, 4:].cpu().numpy()
            results['right_pose'].append(right_pose_pred)
            results['right_trans'].append(right_trans_pred)
            results['left_pose'].append(left_pose_pred)
            results['left_trans'].append(left_trans_pred)
            gts['right_pose'].append(right_pose)
            gts['right_trans'].append(right_trans)
            gts['left_pose'].append(left_pose)
            gts['left_trans'].append(left_trans)
            for i in range(right_pose.shape[0]):
                right_fid = fid(right_pose[i], right_pose_pred[i])
                left_fid = fid(left_pose[i], left_pose_pred[i])
                right_loc_dist = loc_distance(right_trans[i, :, :2], right_trans_pred[i, :, :2])
                left_loc_dist = loc_distance(left_trans[i, :, :2], left_trans_pred[i, :, :2])
                right_MW2 = MW2(right_pose[i], right_pose_pred[i])
                left_MW2 = MW2(left_pose[i], left_pose_pred[i])
                right_smooth = np.mean(np.abs(acceleration(right_pose_pred[i]) - acceleration(right_pose[i])))
                left_smooth = np.mean(np.abs(acceleration(left_pose_pred[i]) - acceleration(left_pose[i])))

                metrics['right']['fid'].append(right_fid)
                metrics['left']['fid'].append(left_fid)
                metrics['right']['dist'].append(right_loc_dist)
                metrics['left']['dist'].append(left_loc_dist)
                metrics['right']['MW'].append(right_MW2)
                metrics['left']['MW'].append(left_MW2)
                metrics['right']['smooth'].append(right_smooth)
                metrics['left']['smooth'].append(left_smooth)

            fid2.push_samples(np.concatenate([right_pose_pred, left_pose_pred], 2),
                                            np.concatenate([right_pose, left_pose], 2))

        for key in metrics['right'].keys():
            metrics['right'][key] = np.mean(metrics['right'][key])
            metrics['left'][key] = np.mean(metrics['left'][key])

            metrics_sum['right'][key] += metrics['right'][key]
            metrics_sum['left'][key] += metrics['left'][key]

        metrics['FID2'], metrics['F_dist'] = fid2.get_scores()

        metrics_sum['FID2'] += metrics['FID2']
        metrics_sum['F_dist'] += metrics['F_dist']

        print(metrics)

    return metrics_sum, metrics, model_path_left, model_path_right


if __name__ == "__main__":
    metrics_sum = {
        'right': {'fid': 0.0, 'MW': 0.0, 'dist': 0.0, 'smooth': 0.0},
        'left': {'fid': 0.0, 'MW': 0.0, 'dist': 0.0, 'smooth': 0.0},
        'FID2': 0.0,
        'F_dist': 0.0
    }
    iter = 10

    best_metrics = None
    best_score = 9999999
    model_path_left = None
    model_path_right = None

    for _ in range(iter):
        metrics_sum, metrics, model_path_left, model_path_right = main(metrics_sum, model_path_left, model_path_right)
        cur_score = 0
        for key, value in metrics.items():
            if isinstance(value, dict):  
                for sub_key in value:
                    norm_value = normalize(value[sub_key], *norm_range[key][sub_key])
                    cur_score += weights[key][sub_key] * norm_value
            else:  
                norm_value = normalize(value, *norm_range[key])
                cur_score += weights[key] * norm_value

        print(f'current score: {cur_score}')
        if cur_score < best_score:
            best_score = cur_score
            best_metrics = metrics

    for key, value in metrics_sum.items():
        if isinstance(value, dict):  
            for sub_key in value:
                value[sub_key] /= iter 
        else:
            metrics_sum[key] /= iter

    print(f'best: {best_metrics}')  
    print(f'average: {metrics_sum}')
