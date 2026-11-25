import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import sys
import math
import pickle
import argparse
import time

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_ntu_act_transition import DatasetNTU
from motion_pred.utils.dataset_grab_action_transition import DatasetGrab
from motion_pred.utils.dataset_humanact12_act_transition import DatasetACT12
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred_tacs import *
from utils.utils import get_dct_matrix

"""dct smoothness + last frame smoothness, with action transition"""


def loss_function(epoch, X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m, gt_labels, y_logits, pred_logits, gt_motion_f, latent_f):
    lambdas = cfg.vae_specs['lambdas']

    if cfg.dataset == 'babel':
        index_used = list(range(30)) + list(range(36,66))
        Y_r = Y_r[:,:,index_used]
        Y = Y[:,:,index_used]
        X = X[:,:,index_used]

    MSE = (Y_r - Y).pow(2).sum(dim=-1).transpose(0, 1)[fn_mask == 1].mean()

    # smoothness
    x = torch.cat([X[-args.N:],Y_r[:args.N]],dim=0).transpose(0,1)
    x_est = torch.matmul(idct_m[None,:,:args.dct_n],torch.matmul(dct_m[None,:args.dct_n],x))
    MSE_v1 = (x_est - x).norm(dim=-1).mean()
    MSE_v2 = (X[-1] - Y_r[0]).norm(dim=-1).mean()
    KLD = 0.5 * torch.sum(plogvar - logvar + (logvar.exp() + (mu - pmu).pow(2)) / plogvar.exp() - 1) / Y.shape[1]

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
    # div =
    if len(lambdas) == 3:
        loss_r = lambdas[0] * (MSE) + lambdas[1] * (MSE_v1) + lambdas[1] * (MSE_v2) + lambdas[2] * (KLD) + y_loss_w * y_loss + pred_loss_w * pred_loss + latent_mse + 2*var_loss
    else:
        loss_r = lambdas[0] * (MSE) + lambdas[1] * (MSE_v1) + lambdas[2] * (MSE_v2) + lambdas[3] * (KLD) + y_loss_w * y_loss + pred_loss_w * pred_loss + latent_mse + 2*var_loss
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
                                                                                        in cfg.vae_specs else 0.08,
                                           is_transi=args.is_transi)

    dct_m, idct_m = get_dct_matrix(args.N*2, is_torch=True,device=device, dtype=dtype)

    for traj_np, label, fn, fn_mask in generator:
        # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        label = tensor(label, device=device, dtype=dtype)
        fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
        fn_mask = tensor(fn_mask[:, t_his:], device=device, dtype=dtype)
        X = traj[:t_his]
        Y = traj[t_his:]
        if cfg.use_ar:
            Y_r, mu, logvar, pmu, plogvar, y_logits, pred_logits, gt_motion_f, latent_f = model(X, Y, label, fn, epoch=epoch)
        else:
            Y_r, mu, logvar, pmu, plogvar, gt_motion_f, latent_f = model(X, Y, label, fn)
            y_logits, pred_logits = None, None
        loss, losses = loss_function(epoch, X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m, label, y_logits, pred_logits, gt_motion_f, latent_f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'train': loss}, epoch)
    # tb_logger.add_scalar('alpha', model.fuse_weight.item(), epoch)


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
                                                                                        in cfg.vae_specs else 0.08)

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
    return train_losses[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=None)
    parser.add_argument('--cfg', default='grab_rnn')
    # parser.add_argument('--cfg_cc', default='grab_act_cc')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--is_other_act', action='store_true', default=False)
    parser.add_argument('--is_transi', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=10) # number of history and future frames for smoothness
    parser.add_argument('--dct_n', type=int, default=5)
    parser.add_argument('--t_pre_extra', type=int, default=0) # extra future poses for stopping
    parser.add_argument('--use_gcn', action='store_true', default=False)
    parser.add_argument('--use_ar', action='store_true', default=False)
    parser.add_argument('--use_tp_acs', action='store_true', default=False,
                        help='Use text-generated class weights for the existing ACS cross-entropy loss')
    parser.add_argument('--use_cosimtpacs', action='store_true', default=False)
    parser.add_argument('--cosin_text', action='store_true', default=False)
    parser.add_argument('--cosin_motion', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--use_dir', action='store_true', default=False)
    parser.add_argument('--use_latent', action='store_true', default=False)
    parser.add_argument('--use_textlabel', action='store_true', default=False)
    parser.add_argument('--decoder_fuse_w', type=float, default=1.0)
    parser.add_argument('--y_w', type=float, default=10.0)
    parser.add_argument('--p_w', type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--use_ddm", action='store_true', default=False)
    parser.add_argument("--text_enc", type=str, default="roberta")
    parser.add_argument("--roberta_type", type=str, default="base")
    parser.add_argument("--clip_type", type=str, default="b16")
    parser.add_argument("--dir_num", type=int, default=30)
    parser.add_argument('--static_ddm', action='store_true', default=False)
    parser.add_argument('--static_ddm_w', type=float, default=0.35)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, exp=args.exp, test=args.test)
    cfg.use_gcn = args.use_gcn
    cfg.use_ar = args.use_ar
    if args.use_tp_acs and (not args.use_ar or not args.use_textlabel):
        parser.error('--use_tp_acs requires both --use_ar and --use_textlabel')
    cfg.vae_specs['use_tp_acs'] = args.use_tp_acs
    cfg.vae_specs['use_cosimtpacs'] = args.use_cosimtpacs
    cfg.vae_specs['cosin_text'] = args.cosin_text
    cfg.vae_specs['cosin_motion'] = args.cosin_motion
    cfg.vae_specs['tau'] = args.tau
    cfg.use_dir = args.use_dir
    cfg.use_latent = args.use_latent
    cfg.use_textlabel = args.use_textlabel
    cfg.vae_specs['decoder_fuse_w'] = args.decoder_fuse_w
    cfg.use_ddm = args.use_ddm
    cfg.vae_specs['text_enc'] = args.text_enc
    cfg.vae_specs['roberta_type'] = args.roberta_type
    cfg.vae_specs['clip_type'] = args.clip_type
    cfg.vae_specs["dir_num"] = args.dir_num
    cfg.vae_specs["static_ddm"] = args.static_ddm
    cfg.vae_specs["static_ddm_w"] = args.static_ddm_w
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
    logger.info(cfg)

    """data"""
    if cfg.dataset == 'grab':
        dataset_cls = DatasetGrab
    elif cfg.dataset == 'ntu':
        dataset_cls = DatasetNTU
    elif cfg.dataset == 'humanact12':
        dataset_cls = DatasetACT12
    elif cfg.dataset == 'babel':
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
    model = get_action_vae_model(cfg, dataset.traj_dim, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))
    criterion = torch.nn.CrossEntropyLoss()
    # cos_sim = nn.CosineSimilarity(dim=(1,2), eps=1e-6)

    """action classifier model"""
    # if args.use_ar:
    #     cfg_classifier = Config(args.cfg_classifier, test=args.test)
    #     model_classifier = get_action_classifier(cfg_classifier, dataset.traj_dim, max_len=dataset.max_len)
    #     # optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    #     # scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    #     print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model_classifier.parameters())) / 1000000.0))
    #     cp_path = cfg_classifier.vae_model_path % 500
    #     print('loading model from checkpoint: %s' % cp_path)
    #     model_cp = pickle.load(open(cp_path, "rb"))
    #     model_classifier.load_state_dict(model_cp['model_dict'])
    #     model_classifier.to(device)
    #     # model_classifier.eval()
    #
    #     model.classifier = model_classifier
    #     for param in model.classifier.parameters():
    #         param.requires_grad = False

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        model.train()
        best_loss = float('inf')
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            test_loss = test(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:

                # with to_cpu(model):
                #     cp_path = cfg.vae_model_path % (i + 1)
                #     model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                #     pickle.dump(model_cp, open(cp_path, 'wb'))
                with to_cpu(model):

                    model_cp = {
                        'model_dict': model.state_dict(),
                        'meta': {
                            'std': dataset.std,
                            'mean': dataset.mean
                        }
                    }

                    # ========= 保存 last =========
                    last_path = cfg.vae_model_path.replace('%04d', 'last')
                    pickle.dump(model_cp, open(last_path, 'wb'))

                    # ========= 保存 best =========
                    if test_loss < best_loss:
                        best_loss = test_loss

                        best_path = cfg.vae_model_path.replace('%04d', 'best')
                        pickle.dump(model_cp, open(best_path, 'wb'))
