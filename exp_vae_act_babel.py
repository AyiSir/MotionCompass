import os
import random
import sys
import math
import pickle
import argparse
import time

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred import *
from utils.utils import get_dct_matrix
import pdb

"""dct smoothness + last frame smoothness, with action transition"""


def loss_function(epoch, X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m, gt_labels, y_logits, pred_logits, gt_motion_f, latent_f):
    lambdas = cfg.vae_specs['lambdas']
    MSE = (Y_r - Y).pow(2).sum(dim=-1).transpose(0, 1)
    MSE[fn_mask == 0] = 0
    MSE = (MSE.sum(dim=1)/(fn_mask.sum(dim=1)+1e-10)).mean()

    # smoothness
    x = torch.cat([X[-args.N:],Y_r[:args.N]],dim=0).transpose(0,1)
    x_est = torch.matmul(idct_m[None,:,:args.dct_n],torch.matmul(dct_m[None,:args.dct_n],x))
    MSE_v1 = (x_est - x).norm(dim=-1).mean()
    MSE_v2 = (X[-1] - Y_r[0]).norm(dim=-1).mean()
    KLD = 0.5 * torch.sum(plogvar - logvar + (logvar.exp() + (mu - pmu).pow(2)) / (plogvar.exp()+1e-10) - 1) / Y.shape[1]

    # crossentropy
    if pred_logits is not None:
        gt_labels = torch.argmax(gt_labels, dim=-1)
        if y_logits is not None:
            y_loss = criterion(y_logits, gt_labels)
        else:
            y_loss = torch.tensor(0).float()
        pred_loss = criterion(pred_logits, gt_labels)
    else:
        y_loss = torch.tensor(0).float()
        pred_loss = torch.tensor(0).float()

    if gt_motion_f is not None and latent_f is not None:
        # # latent_mse = (gt_motion_f - latent_f).pow(2).sum(dim=-1).mean()
        # latent_mse = (gt_motion_f - latent_f).norm(dim=-1).mean()
        # var = torch.var(gt_motion_f, dim=0, unbiased=False)
        # std = torch.sqrt(var + 1e-8)
        # loss_vec = torch.relu(1 - std)
        # var_loss = loss_vec.mean()
        # _, bs, _ = gt_motion_f.shape
        # similarity = torch.cosine_similarity(gt_motion_f.permute(1,0,2).contiguous().reshape(bs,-1), latent_f.permute(1,0,2).contiguous().reshape(bs,-1), dim=1)
        # similarity = similarity.mean()
        gt_motion_f = nn.functional.normalize(gt_motion_f, dim=-1)
        latent_f = nn.functional.normalize(latent_f, dim=-1)
        similarity = torch.cosine_similarity(gt_motion_f, latent_f, dim=-1)  # [B]
        var_loss = 1.0 - similarity.mean()
        latent_mse = torch.tensor(0).float()
        # sim = cos_sim(a, b)
    else:
        latent_mse = torch.tensor(0).float()
        var_loss = torch.tensor(0).float()

    if epoch < 999:
        # Phase 1: 语义注入期
        y_loss_w = args.y_w  # z
        pred_loss_w = args.p_w  # h

    # elif epoch < 100:
    #     # Phase 2: 语义-生成平衡期（线性衰减）
    #     # z: 15 -> 5
    #     y_loss_w = 15.0 - (epoch - 20) * (10.0 / 80.0)
    #     # h: 5 -> 0
    #     pred_loss_w = 5.0 - (epoch - 20) * (5.0 / 80.0)

    else:
        # Phase 3: 纯生成精修期
        y_loss_w = 0  # 或者 0.0，如果你想完全放飞 latent
        pred_loss_w = 1
    # regularization
    if len(lambdas) == 3:
        loss_r = lambdas[0] * MSE + lambdas[1] * MSE_v1 + lambdas[1] * MSE_v2 + lambdas[2] * KLD + y_loss_w * y_loss + pred_loss_w * pred_loss + latent_mse + 2*var_loss
    else:
        loss_r = lambdas[0] * MSE + lambdas[1] * MSE_v1 + lambdas[2] * MSE_v2 + lambdas[3] * KLD + y_loss_w * y_loss + pred_loss_w * pred_loss + latent_mse + 2*var_loss
    # loss_r = MSE
    return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v1.item(), MSE_v2.item() , KLD.item(), y_loss.item(), pred_loss.item(), latent_mse.item(), var_loss.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    train_grad = 0
    loss_names = ['TOTAL', 'MSE', 'DCT_smooth', 'Lastframe_smooth', 'KLD', 'y_loss', 'pred_loss', 'latent_mse', 'var_loss']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=args.is_other_act, t_pre_extra=args.t_pre_extra,
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 25,
                                           is_transi=args.is_transi, n_others=args.n_other,
                                           others_all_act=cfg.vae_specs.get('others_all_act',False))

    dct_m, idct_m = get_dct_matrix(args.N*2, is_torch=True,device=device,dtype=dtype)

    for traj_np, label, fn, fn_mask in generator:
        # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        label = tensor(label, device=device, dtype=dtype)
        fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
        fn_mask = tensor(fn_mask[:, t_his:], device=device, dtype=dtype)
        X = traj[:t_his]
        Y = traj[t_his:]

        if cfg.dataset == 'babel':
            index_used = list(range(30)) + list(range(36, 66))
            Y = Y[:, :, index_used]
            X = X[:, :, index_used]

        if cfg.use_ar:
            Y_r, mu, logvar, pmu, plogvar, y_logits, pred_logits, gt_motion_f, latent_f = model(X, Y, label, fn, epoch=epoch)
        else:
            Y_r, mu, logvar, pmu, plogvar, gt_motion_f, latent_f = model(X, Y, label, fn)
            y_logits, pred_logits = None, None
        loss, losses = loss_function(epoch, X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m, label, y_logits, pred_logits, gt_motion_f, latent_f)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=200)
        if torch.isnan(loss) or torch.isinf(loss) or loss > 100000 or \
                torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100000:
            continue
            # pdb.set_trace()
        train_grad += grad_norm
        optimizer.step()
        train_losses += losses
        total_num_sample += 1
        # print(total_num_sample)

    scheduler.step()
    dt = time.time() - t_s
    if not(type(train_losses) == np.ndarray):
        train_losses = np.zeros_like(losses)
    train_losses /= (total_num_sample+1e-10)
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f} total samp: {:d}'.format(epoch, dt,
                                                                                     losses_str, lr,
                                                                                     total_num_sample))
    tb_logger.add_scalar('train_grad', train_grad / (total_num_sample+1e-10), epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'train': loss}, epoch)


def test(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    y_acc = 0.
    pred_acc = 0.
    loss_names = ['TOTAL', 'MSE', 'DCT_smooth', 'Lastframe_smooth', 'KLD', 'y_loss', 'pred_loss', 'latent_mse', 'var_loss']
    generator = dataset_test.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=args.is_other_act, t_pre_extra=args.t_pre_extra,
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 25,
                                                n_others=args.n_other,
                                           others_all_act=cfg.vae_specs['others_all_act'])


    dct_m, idct_m = get_dct_matrix(args.N*2, is_torch=True,device=device)
    with torch.no_grad():
        for traj_np, label, fn, fn_mask in generator:
            # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
            traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            label = tensor(label, device=device, dtype=dtype)
            fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
            fn_mask = tensor(fn_mask[:, t_his:], device=device, dtype=dtype)
            X = traj[:t_his]
            Y = traj[t_his:]

            if cfg.dataset == 'babel':
                index_used = list(range(30)) + list(range(36, 66))
                Y = Y[:, :, index_used]
                X = X[:, :, index_used]

            if cfg.use_ar:
                Y_r, mu, logvar, pmu, plogvar, y_logits, pred_logits, gt_motion_f, latent_f = model(X, Y, label, fn, epoch=epoch)
            else:
                Y_r, mu, logvar, pmu, plogvar, gt_motion_f, latent_f = model(X, Y, label, fn)
                y_logits, pred_logits = None, None
            loss, losses = loss_function(epoch, X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m, label, y_logits, pred_logits, gt_motion_f, latent_f)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if cfg.use_ar:
                label = torch.argmax(label, dim=1)
                if y_logits is not None:
                    y_label = torch.argmax(y_logits, dim=1)
                    y_acc += (y_label == label).float().mean()
                else:
                    y_acc= 0.0
                pred_label = torch.argmax(pred_logits, dim=1)
                pred_acc += (pred_label == label).float().mean()
            train_losses += losses
            total_num_sample += 1

    # scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    y_acc /= total_num_sample
    pred_acc /= total_num_sample
    # lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    if cfg.use_ar:
        logger.info('====> Epoch Test: {} Time: {:.2f} {},{}: {:.4f},{}: {:.4f}'.format(epoch, dt, losses_str, "y acc", y_acc, "pred_acc", pred_acc))
    else:
        logger.info('====> Epoch Test: {} Time: {:.2f} {}'.format(epoch, dt, losses_str))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'test': loss}, epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=None)
    parser.add_argument('--cfg', default='babel_v3_5_4_t1_50_10_10_1_others_all_act_float')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--is_other_act', action='store_true', default=False)
    parser.add_argument('--n_other', type=int, default=1)
    parser.add_argument('--is_transi', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=10) # number of history and future frames for smoothness
    parser.add_argument('--dct_n', type=int, default=5)
    parser.add_argument('--t_pre_extra', type=int, default=0) # extra future poses for stopping
    parser.add_argument('--use_gcn', action='store_true', default=False)
    parser.add_argument('--use_ar', action='store_true', default=False)
    parser.add_argument('--use_dir', action='store_true', default=False)
    parser.add_argument('--use_latent', action='store_true', default=False)
    parser.add_argument('--use_textlabel', action='store_true', default=False)
    parser.add_argument('--decoder_fuse_w', type=float, default=1.0)
    parser.add_argument('--y_w', type=int, default=5)
    parser.add_argument('--p_w', type=int, default=5)
    parser.add_argument("--use_ddm", action='store_true', default=False)
    parser.add_argument("--dir_num", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, exp=args.exp, test=args.test)
    cfg.use_gcn = args.use_gcn
    cfg.use_ar = args.use_ar
    cfg.use_dir = args.use_dir
    cfg.use_latent = args.use_latent
    cfg.use_textlabel = args.use_textlabel
    cfg.vae_specs['decoder_fuse_w'] = args.decoder_fuse_w
    cfg.use_ddm = args.use_ddm
    cfg.vae_specs["dir_num"] = args.dir_num
    cfg.vae_lr = args.lr
    # cfg.exp = args.exp
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if 'smooth_N' in cfg.vae_specs:
        args.N = cfg.vae_specs['smooth_N']
    if 'dct_n' in cfg.vae_specs:
        args.dct_n = cfg.vae_specs['dct_n']
    if 't_pre_extra' in cfg.vae_specs:
        args.t_pre_extra = cfg.vae_specs['t_pre_extra']
    if 'is_other_act' in cfg.vae_specs:
        args.is_other_act = cfg.vae_specs['is_other_act']
    cfg.vae_specs['others_all_act'] = cfg.vae_specs.get('others_all_act',False)
    logger.info(cfg)

    """data"""
    if cfg.dataset == 'babel':
        dataset_cls = DatasetBabel
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False)
    dataset_test = dataset_cls('test', t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False)
    cfg.train_actnames = dataset.act_name
    cfg.test_actnames = dataset_test.act_name
    logger.info(f'Training data sequences {dataset.data_len:d}.')
    logger.info(f'Testing data sequences {dataset_test.data_len:d}.')
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    model = get_action_vae_model(cfg, 60, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))
    criterion = torch.nn.CrossEntropyLoss()

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        # model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            model.train()
            train(i)
            # model.eval()
            # test(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))
