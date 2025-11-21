import argparse
import copy
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import numpy as np
import torch
import torch.utils.data as data
from scipy.signal import savgol_filter
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets.PianoPose import PianoPose
from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D
from models.utils import velocity_loss
from datasets.show import render_result
from models.BalancedDataParallel import BalancedDataParallel
import copy
DEBUG = 0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bs_dim", type=int, default=122, help='number of blendshapes:122')
    parser.add_argument("--feature_dim", type=int, default=832, help='number of feature dim')
    parser.add_argument("--period", type=int, default=30, help='number of period')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max sequence length')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu0_bs", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=16)
    # additional
    parser.add_argument("--experiment_name", type=str, default="test")
    # dataset
    parser.add_argument("--data_root", type=str, default='/hdd/data3/piano-bilibili')
    parser.add_argument("--preload", action='store_true')
    parser.add_argument("--tiny", action='store_true')
    parser.add_argument("--adjust", action='store_true')
    parser.add_argument("--use_midiguide", action='store_true')
    parser.add_argument("--is_random", action='store_true')
    parser.add_argument("--return_beta", action='store_true')
    parser.add_argument('--up_list', nargs='+', default=[])
    # model
    parser.add_argument("--continue_train", action='store_true')
    parser.add_argument("--piano2posi_path", type=str, default="logs/piano2posi_tf_wgrad")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--unet_dim", type=int, default=128)
    parser.add_argument("--xyz_guide", action='store_true')
    parser.add_argument("--remap_noise", type=bool, default=True)
    parser.add_argument("--hidden_type", choices=['audio_f', 'hidden_f', 'both'], default='audio_f')
    parser.add_argument("--latest_layer", choices=['sigmoid', 'tanh', 'none'], default='tanh')
    parser.add_argument("--encoder_type", choices=['none', 'transformer', 'mamba'], default='none')
    parser.add_argument("--num_layer", type=int, default=16)
    # train
    parser.add_argument("--loss_mode", choices=['naive_l2', 'naive_l1'], default='naive_l1')
    parser.add_argument("--weight_rec", type=float, default=1.)
    parser.add_argument("--weight_vel", type=float, default=1.)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--train_sec", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--check_val_every_n_iteration", type=int, default=500)
    parser.add_argument("--save_every_n_iteration", type=int, default=500)
    parser.add_argument("--logdir", type=str, default="logs")

    parser.add_argument("--fusion", type=int, default=0)
    parser.add_argument("--obj", choices=['pred_noise', 'pred_x0', 'pred_v'], default='pred_v')
    args = parser.parse_args()
    return args

def main():
    # get args
    args = get_args()
    os.makedirs(args.logdir + '/' + args.experiment_name, exist_ok=True)
    out_dire = args.logdir + '/' + args.experiment_name # + '/' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    os.makedirs(out_dire, exist_ok=True)
    with open(out_dire + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # get dataloader
    train_loader = data.DataLoader(
        PianoPose(args=args, phase='train', tiny=args.tiny),
        batch_size=args.batch_size, shuffle=True, num_workers=16,
        drop_last=True)
    valid_loader = data.DataLoader(
        PianoPose(args=args, phase='valid', tiny=True),
        batch_size=args.valid_batch_size,
        drop_last=False)

    args_old = copy.copy(args)
    with open(args.piano2posi_path + '/args.txt', 'r') as f:
        dic = json.load(f)
        dic['hidden_type'] = args.hidden_type
        args_old.__dict__ = dic
    with open(out_dire + '/args_posi.txt', 'w') as f:
        json.dump(args_old.__dict__, f, indent=2)
    # load model
    piano2posi_left = Piano2Posi(args_old)
    piano2posi_right = copy.deepcopy(piano2posi_left)

    checkpoint_path_left = args.piano2posi_path + '/' + sorted([i for i in os.listdir(args.piano2posi_path) if '.ckpt' in i and 'left' in i])[-1]
    print('Piano2posi left load from:', checkpoint_path_left)
    checkpoint_path_right = args.piano2posi_path + '/' + sorted([i for i in os.listdir(args.piano2posi_path) if '.ckpt' in i and 'right' in i])[-1]
    print('Piano2posi right load from:', checkpoint_path_right)
    checkpoint_left = torch.load(checkpoint_path_left)
    checkpoint_right = torch.load(checkpoint_path_right)

    piano2posi_left.load_state_dict(checkpoint_left['state_dict'])
    piano2posi_right.load_state_dict(checkpoint_right['state_dict'])
    piano2posi_left.cpu()
    piano2posi_right.cpu()
    del checkpoint_left, checkpoint_right

    if args_old.hidden_type == 'audio_f':
        cond_dim = 768 if 'base' in args_old.wav2vec_path else 1024
    elif args_old.hidden_type == 'hidden_f':
        cond_dim = args_old.feature_dim
    elif args_old.hidden_type == 'both':
        cond_dim = args_old.feature_dim + (768 if 'base' in args_old.wav2vec_path else 1024)


    # get model
    model = Unet1D(
        dim=args.unet_dim,
        dim_mults=(1, 2, 4, 8),
        # channels=args.bs_dim,
        channels=args.bs_dim // 2,
        remap_noise=args.remap_noise if 'remap_noise' in args.__dict__ else True,
        condition=True,
        guide=args.xyz_guide,
        guide_dim=6 if args.xyz_guide else 0,
        condition_dim=cond_dim,
        encoder_type=args.encoder_type,
        num_layer=args.num_layer,
        feature_fusion=args.fusion,
    )

    diffusion_left = GaussianDiffusion1D_piano2pose(
        model,
        piano2posi_left,
        seq_length=args.train_sec * 30,
        timesteps=args.timesteps,
        objective=args.obj,
        train_acceLoss_weight = 0,
        train_velLoss_weight = 0,
    )
    
    diffusion_right = GaussianDiffusion1D_piano2pose(
        copy.deepcopy(model),
        piano2posi_right,
        seq_length=args.train_sec * 30,
        timesteps=args.timesteps,
        objective=args.obj,
        train_acceLoss_weight = 0,
        train_velLoss_weight = 0,
    )   

    diffusion_left = diffusion_left.cuda()
    diffusion_right = diffusion_right.cuda()
    if args.continue_train:
        model_path_left = sorted([i for i in os.listdir(out_dire) if 'ckpt' in i and 'left' in i], key=lambda x: int(x.split('-')[2].split('=')[1]))[-1]
        model_path_right = sorted([i for i in os.listdir(out_dire) if 'ckpt' in i and 'right' in i], key=lambda x: int(x.split('-')[2].split('=')[1]))[-1]
        print('left hand load from:', model_path_left)
        print('right hand load from:', model_path_right)
        state_dict_left = torch.load(out_dire + '/' + model_path_left)['state_dict']
        diffusion_left.load_state_dict(state_dict_left)
        state_dict_right = torch.load(out_dire + '/' + model_path_right)['state_dict']
        diffusion_right.load_state_dict(state_dict_right)
        del state_dict_left, state_dict_right

        start_iter = int(model_path_left.split('-')[2].split('=')[1]) + 1
    else:
        start_iter = 0

    optimizer_left = optim.Adam(diffusion_left.parameters(), lr=args.lr)
    optimizer_right = optim.Adam(diffusion_right.parameters(), lr=args.lr)
    tb_writer = SummaryWriter(out_dire)

    print('GPU number: {}'.format(torch.cuda.device_count()))
    diffusion_left.to('cuda')
    diffusion_right.to('cuda')
    if torch.cuda.device_count() == 1:
        diffusion_left = torch.nn.DataParallel(diffusion_left)
        diffusion_right = torch.nn.DataParallel(diffusion_right)
    else:
        diffusion_left = BalancedDataParallel(args.gpu0_bs, diffusion_left, dim=0)
        diffusion_right = BalancedDataParallel(args.gpu0_bs, diffusion_right, dim=0)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {trainable_params / 1e6}M')

    pbar = tqdm(range(start_iter, args.iterations))
    iter_train = iter(train_loader)
    scale = torch.tensor([1.5, 1.5, 25]).cuda()
    last_checkpoint_path_left = None
    last_checkpoint_path_right = None
    best_score = 999999
    for iteration in pbar:
        try:
            batch = next(iter_train)
        except StopIteration:
            iter_train = iter(train_loader)
            batch = next(iter_train)

        for key in batch.keys():
            batch[key] = batch[key].cuda()

        audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']

        gt_left = left_pose[:, :, 4:].permute(0, 2, 1) / np.pi          # B, 48, N
        gt_right = right_pose[:, :, 4:].permute(0, 2, 1) / np.pi        # B, 48, N

        guide_left, cond_left = diffusion_left.module.getGuideAndCond(gt_left, audio=audio)
        guide_right, cond_right = diffusion_right.module.getGuideAndCond(gt_right, audio=audio)

        guide_left_total = torch.cat([guide_left, guide_right], 2)
        guide_right_total = torch.cat([guide_right, guide_left], 2)

        b, c, n, device = *gt_left.shape, gt_left.device
        t = torch.randint(0, args.timesteps, (b,), device=device).long()
        down_output_left, noise_left, x_t_left = diffusion_left.module.forward_down(gt_left, t, guide_left_total, cond_left)
        down_output_right, noise_right, x_t_right = diffusion_right.module.forward_down(gt_right, t, guide_right_total, cond_right)

        x_down_left, r_left, t_emb_left, h_left = down_output_left
        x_down_right, r_right, t_emb_right, h_right = down_output_right

        loss_left = diffusion_left(x_down_left, x_down_right, r_left, t, t_emb_left, h_left, noise_left, gt_left, x_t_left)
        loss_right = diffusion_right(x_down_right, x_down_left, r_right, t, t_emb_right, h_right, noise_right, gt_right, x_t_right)

        optimizer_left.zero_grad()
        optimizer_right.zero_grad()
        loss = loss_left + loss_right
        loss.backward()
        optimizer_left.step()
        optimizer_right.step()

        des = 'loss_left:%.4f, loss_right:%.4f' % (loss_left.item(), loss_right.item())
        tb_writer.add_scalar("train_loss_patches/loss_left", loss_left.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/loss_right", loss_right.item(), iteration)
        pbar.set_description(des)

        if (iteration % args.check_val_every_n_iteration == 0 and iteration != 0) or iteration == args.iterations - 1:
            rec_losss = []
            vel_losss = []
            losss = []
            with torch.no_grad():
                for v_idx, batch in enumerate(valid_loader):
                    for key in batch.keys():
                        batch[key] = batch[key].cuda()

                    audio, right_pose, left_pose = batch['audio'], batch['right'], batch['left']

                    frame_num = left_pose.shape[1]
                    gt = torch.cat([right_pose[:, :, 4:], left_pose[:, :, 4:]], 2) / np.pi  # B, 96, N
                    gt_left = left_pose[:, :, 4:].permute(0, 2, 1) / np.pi          # B, 48, N
                    gt_right = right_pose[:, :, 4:].permute(0, 2, 1) / np.pi        # B, 48, N

                    guide_left, cond_left = diffusion_left.module.getGuideAndCond(gt_left, audio=audio)
                    guide_right, cond_right = diffusion_right.module.getGuideAndCond(gt_right, audio=audio)

                    guide_left_total = torch.cat([guide_left, guide_right], 2)
                    guide_right_total = torch.cat([guide_right, guide_left], 2)

                    pose_hat_left, pose_hat_right = diffusion_left.module.sample(guide_left_total, cond_left, guide_right_total, cond_right, diffusion_right.module.model, gt_left.shape[0])

                    pose_hat = torch.cat([pose_hat_right, pose_hat_left], 1)    # B, 96, N
                    pose_hat = pose_hat.permute(0, 2, 1)
                    guide = torch.cat([guide_right, guide_left], 2)
                    # guide = guide.permute(0, 2, 1)
                    if v_idx == 0:
                        prediction = pose_hat[0].detach().cpu().numpy() * np.pi
                        guide = (guide * scale.repeat(2))[0].cpu().numpy()
                        for i in range(prediction.shape[1]):
                            prediction[:, i] = savgol_filter(prediction[:, i], 5, 2)
                        render_result(out_dire + '/result-%d.mp4' % iteration, audio[0].cpu().numpy(),# prediction[:, :51], prediction[:, 51:])
                                      np.concatenate([guide[:, :3], prediction[:, :48]], 1),
                                      np.concatenate([guide[:, 3:], prediction[:, 48:]], 1))
                        render_result(out_dire + '/gt-%d.mp4' % iteration, audio[0].cpu().numpy(),
                                      # prediction[:, :51], prediction[:, 51:])
                                      right_pose[0, :, 1:52].cpu().numpy(),
                                      left_pose[0, :, 1:52].cpu().numpy())
                        print(out_dire)

                    right_hat = pose_hat[:, :, :args.bs_dim // 2]
                    left_hat = pose_hat[:, :, args.bs_dim // 2:]

                    pose_hat_sel = torch.cat([right_hat[right_pose[:, :, 0] == 1], left_hat[left_pose[:, :, 0] == 1]],
                                             0)
                    pose_gt_sel = torch.cat([right_pose[right_pose[:, :, 0] == 1], left_pose[left_pose[:, :, 0] == 1]],
                                            0)
                    pose_gt_sel = pose_gt_sel[:, 4:] / np.pi

                    # get loss
                    if args.loss_mode.startswith('naive'):
                        if args.loss_mode == 'naive_l1':
                            func = torch.nn.functional.l1_loss
                        elif args.loss_mode == 'naive_l2':
                            func = torch.nn.functional.mse_loss
                        rec_loss = func(pose_gt_sel, pose_hat_sel)
                        vel_loss = velocity_loss(gt, pose_hat, func)
                    loss = args.weight_rec * rec_loss + args.weight_vel * vel_loss

                    rec_losss.append(rec_loss.item())
                    vel_losss.append(vel_loss.item())
                    losss.append(loss.item())
            print('Val Iter %d: loss: %.4f, rec_loss: %.4f, vel_loss: %.4f' % (iteration, np.mean(losss), np.mean(rec_losss), np.mean(vel_losss)))
            tb_writer.add_scalar("eval/rec", np.mean(rec_losss), iteration)
            tb_writer.add_scalar("eval/vel", np.mean(vel_losss), iteration)
            if iteration % args.save_every_n_iteration == 0  or iteration == args.iterations - 1:
                if np.mean(losss) < best_score:
                    best_score = np.mean(losss)
                    if last_checkpoint_path_left is not None:
                        os.system(f'rm {last_checkpoint_path_left}')
                        os.system(f'rm {last_checkpoint_path_right}')
                    last_checkpoint_path_left = out_dire + '/piano2pose-left-iter=%d-val_loss=%.16f.ckpt' % (iteration, best_score)
                    torch.save({'state_dict': diffusion_left.module.state_dict(), 'hyper_parameters': args},
                            last_checkpoint_path_left)
                    last_checkpoint_path_right = out_dire + '/piano2pose-right-iter=%d-val_loss=%.16f.ckpt' % (iteration, best_score)
                    torch.save({'state_dict': diffusion_right.module.state_dict(), 'hyper_parameters': args},
                            last_checkpoint_path_right)


if __name__ == "__main__":
    main()
