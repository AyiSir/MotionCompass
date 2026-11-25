import math

import torch
import numpy as np
from torch import nn, sqrt
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from models.mlp import MLP
from models.rnn import RNN
from utils.torch import *
from models import transformer_vae
from tools.skeleton import Skeleton
from models.st_gcn import st_gcn
from models.attention import CrossAttentionFusion
import models.clip as clip
from easydict import EasyDict as edict

def build_derangement(num_classes, seed=2026):
    """生成无固定点置换，保证 permutation[i] != i。"""
    if num_classes <= 1:
        raise ValueError("Derangement requires num_classes > 1.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    original = torch.arange(num_classes)
    while True:
        permutation = torch.randperm(num_classes, generator=generator)
        if torch.all(permutation != original):
            return permutation

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        if torch.get_default_dtype() != torch.float32:
            self.model = model.double()
        else:
            self.model = model.float()

    def forward(self,text):
        return self.model.encode_text(text)


class MotionCompass(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """
    def __init__(self, nx, ny, nz, horizon, specs, data=None, gcn=False, ar=False, dir=False, latent=False, text_label=False, ddm=False, actnames=None):
        super(MotionCompass, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        # self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)
        self.is_layernorm = is_layernorm = specs.get('is_layernorm', False)
        self.is_bn = is_bn = specs.get('is_bn', True)
        self.specs = specs

        ##
        self.gcn = gcn
        if self.gcn:
            self.use_gcn_decoder = True
        else:
            self.use_gcn_decoder = False
        self.data = data
        if self.data == 'ntu':
            self.joints = 24
        if self.data == 'ntu':
            self.joints = 24
        if self.data == 'ntu':
            self.joints = 24
        if self.data == 'ntu':
            self.joints = 24
        self.actnames = actnames
        self.ar = ar
        self.dir = dir
        self.latent = latent
        self.text_label = text_label
        self.ddm = ddm

        # encode
        ## gcn test (no used)
        if self.gcn:
            self.x_gcn = gcn_encoder(nx, nh_rnn, data=self.data, joint=self.joints)
            if self.use_gcn_decoder:
                self.y_gcn = gcn_decoder(nh_rnn, nx, data=self.data, joint=self.joints)

        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=False, cell_type=rnn_type,is_layernorm=is_layernorm)
        # self.e_rnn.set_mode('step')
        self.e_mlp = MLP(3 * nh_rnn, nh_mlp, is_bn=is_bn)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)

        # prior
        self.p_act_mlp = nn.Linear(n_action, nh_rnn)
        self.p_mlp = MLP(2 * nh_rnn, nh_mlp, is_bn=is_bn)
        self.p_mu = nn.Linear(self.p_mlp.out_dim, nz)
        self.p_logvar = nn.Linear(self.p_mlp.out_dim, nz)

        if self.text_label:
            self.e_mlp_text = MLP(3 * nh_rnn, nh_mlp, is_bn=is_bn)
            self.e_mu_text = nn.Linear(self.e_mlp.out_dim, nz)
            self.e_logvar_text = nn.Linear(self.e_mlp.out_dim, nz)

            # prior
            self.p_mlp_text = MLP(2 * nh_rnn, nh_mlp, is_bn=is_bn)
            self.p_mu_text = nn.Linear(self.p_mlp.out_dim, nz)
            self.p_logvar_text= nn.Linear(self.p_mlp.out_dim, nz)

        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        if self.dir:
            self.d_rnn = RNN(ny + nz + nh_rnn + nh_rnn, nh_rnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        else:
            self.d_rnn = RNN(ny + nz + nh_rnn + nh_rnn, nh_rnn, cell_type=rnn_type, is_layernorm=is_layernorm)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')
        if self.text_label:
            if self.dir:
                self.d_rnn_text = RNN(ny + nz + nh_rnn + nh_rnn, nh_rnn, cell_type=rnn_type, is_layernorm=is_layernorm)
            else:
                self.d_rnn_text = RNN(ny + nz + nh_rnn + nh_rnn, nh_rnn, cell_type=rnn_type, is_layernorm=is_layernorm)
            self.d_mlp_text = MLP(nh_rnn, nh_mlp)
            self.d_out_text = nn.Linear(self.d_mlp.out_dim, ny)
            self.d_rnn_text.set_mode('step')
            self.fuse_weight = nn.Parameter(torch.tensor(0.5))


        ## ACS part
        if self.ar:
            self.pred_head = nn.Linear(nh_rnn, self.n_action)
            self.drop = nn.Dropout(0.1)

        ## direction part
        if self.dir:
            self.vec_dim = 256
            # self.vec_num = self.specs['dir_num']
            self.vec_num = 30
            print(self.vec_num)
            self.dirLinear1 = nn.Linear(128, self.vec_num)
            self.dirLinear2 = nn.Linear(self.vec_num, self.vec_num)
            self.dirLinearAfter = nn.Linear(256, nh_rnn)
            self.direction = Direction(self.vec_num, self.vec_dim)

            self.dir_class = nn.Linear(self.vec_dim, self.vec_num)
            ## DDM part
            if self.ddm:
                self.z_linear = nn.Linear(self.nz, nh_rnn)
                self.y_p_mlp = MLP(nx, [200, nh_rnn])

            self.time_mlp = MLP(30,[128,30])
            self.vec_mlp = MLP(256, [128,128,72])
            self.vec_out = nn.Linear(self.vec_mlp.out_dim, ny)

            # self.static_ddm = self.specs['static_ddm']
            self.static_ddm = False
            # self.ddm_w = self.specs['static_ddm_w']
            self.static_ddm_w = 0.5


        ## text part
        if self.text_label:
            # self.textencoder = "clip"
            self.textencoder = specs.get('text_enc', "roberta")
            self.roberta_type = specs.get('roberta_type', "base")
            self.clip_type = specs.get('clip_type', "b16")
            if self.textencoder =="roberta":
                tokenlen = 20
                if self.roberta_type == "base":
                    path = "FacebookAI/roberta-base"
                elif self.roberta_type == "large":
                    path = "FacebookAI/roberta-large"
                self.tokenizer = AutoTokenizer.from_pretrained(path)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.transformer = AutoModel.from_pretrained(path, ignore_mismatched_sizes=True).cuda()
                self.transformer.config.pad_token_id = self.transformer.config.eos_token_id
                for param in self.transformer.parameters():
                    param.requires_grad = False

                prompt=[]
                for act in actnames:
                    prompt.append(f"The future action is: \"{act}\".")
                    #prompt.append(f"{act}")
                    # prompt.append(f"The current action will smoothly transition to the next action \"{act}\".")
                    # "The person is performing a {act} motion with dynamic movement."
                ## new
                des_token = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenlen).to("cuda")
                self.text_label_feats = self.transformer(**des_token).last_hidden_state

                # self.text_label_feats = self.text_label_feats.squeeze()
                self.textLinear = nn.Linear(self.text_label_feats.shape[-1], 128)
                # self.tokenLinear = nn.Linear(tokenlen, 1)
                # self.text_mlp = MLP(self.text_label_feats.shape[-1], [256, 128])
                self.text_mlp = MLP(128, [128, 128])
                # self.text_mlp = MLP(nh_rnn + nh_rnn, [128, 128])
                # self.textCrossAtt = CrossAttentionFusion(input_dim=[128, 768], hidden_dim=128)
                self.text_norm = nn.LayerNorm(128)
            elif self.textencoder == "clip":
                if self.clip_type == "b16":
                    path = 'ViT-B/16'
                elif self.clip_type == "b32":
                    path = 'ViT-B/32'
                elif self.clip_type == "l14":
                    path = 'ViT-L/14'
                model_, preprocess = clip.load(path, "cuda" if torch.cuda.is_available() else "cpu")
                self.text_encoder = TextCLIP(model_).cuda()
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                prompt = []
                for act in actnames:
                    prompt.append(f"The future action is: \"{act}\".")
                    # prompt.append(f"The current action will smoothly transition to the next action \"{act}\".")
                des_token = clip.tokenize(prompt).cuda()
                self.text_label_feats = self.text_encoder(des_token).type(torch.get_default_dtype())
                # self.textLinear = nn.Linear(self.text_label_feats.shape[-1], 128)
                self.text_mlp = MLP(self.text_label_feats.shape[-1], [256, 128])
                # self.textCrossAtt = CrossAttentionFusion(input_dim=[128, 768], hidden_dim=128)
            # elif self.textencoder == 'learned_id':
            #     query_dim = 128
            #     # 与 RoBERTa-base 的 Linear(768, 128) 匹配。
            #     reference_dim = 768
            #
            #     match_params = True
            #
            #     if match_params:
            #         reference_params = (
            #                 reference_dim * query_dim + query_dim
            #         )
            #
            #         learned_id_dim = int(round(
            #             (reference_params - query_dim)
            #             / float(self.n_action + query_dim)
            #         ))
            #     else:
            #         # 简化版：直接学习128维类别查询。
            #         learned_id_dim = query_dim
            #
            #     self.learned_id_dim = learned_id_dim
            #
            #     self.learned_id_embedding = nn.Embedding(
            #         self.n_action,
            #         learned_id_dim
            #     )
            #
            #     if learned_id_dim == query_dim:
            #         self.learned_id_proj = nn.Identity()
            #     else:
            #         self.learned_id_proj = nn.Linear(
            #             learned_id_dim,
            #             query_dim
            #         )
            #
            #     learned_id_params = (
            #             self.n_action * learned_id_dim
            #             + (
            #                 learned_id_dim * query_dim + query_dim
            #                 if learned_id_dim != query_dim
            #                 else 0
            #             )
            #     )
            #
            #     print(
            #         '[Learned-ID] n_action={}, id_dim={}, '
            #         'query_dim={}, active_params={}'.format(
            #             self.n_action,
            #             learned_id_dim,
            #             query_dim,
            #             learned_id_params
            #         )
            #     )

    #     # random test
    #     permute_text_label = False
    #     text_perm_seed = 2026
    #     self.permute_text_label = permute_text_label
    #     self.text_perm_seed = text_perm_seed
    #
    #     num_text_labels = self.text_label_feats.shape[0]
    #
    #     text_label_permutation = build_derangement(
    #         num_classes=num_text_labels,
    #         seed=self.text_perm_seed
    #     )
    #
    #     self.register_buffer(
    #         "text_label_permutation",
    #         text_label_permutation,
    #         persistent=False
    #     )
    #
    #     if self.permute_text_label:
    #         print(
    #             "[Text-label permutation] seed = {}, mapping = {}".format(
    #                 self.text_perm_seed,
    #                 self.text_label_permutation.tolist()
    #             )
    #         )
    #
    # def get_action_query(self, act):
    #     if not self.text_label:
    #         return None
    #
    #     labels = act.argmax(dim=1).long()
    #
    #     if self.textencoder == 'learned_id':
    #         h_text = self.learned_id_embedding(labels)
    #         h_text = self.learned_id_proj(h_text)
    #         return h_text
    #     text_labels = labels
    #     if getattr(self, 'permute_text_label', False):
    #         permutation = self.text_label_permutation.to(
    #             labels.device
    #         )
    #         text_labels = permutation[labels]
    #
    #     if self.textencoder == 'roberta':
    #         h_text = self.text_label_feats[text_labels]
    #         h_text = torch.mean(
    #             self.textLinear(h_text),
    #             dim=1
    #         )
    #         return h_text
    #
    #     if self.textencoder == 'clip':
    #         h_text = self.text_label_feats[text_labels]
    #         h_text = self.text_mlp(h_text)
    #         return h_text
    #
    #     raise ValueError(
    #         'Unsupported text_enc: {}'.format(
    #             self.textencoder
    #         )
    #     )

    def encode_x(self, x):
        if self.gcn:
            x0 = x.permute(1, 2, 0).contiguous()
            h_x1 = self.x_gcn(x0)[-1]
            # h_x = h_x1
            if self.x_birnn:
                h_x2 = self.x_rnn(x).mean(dim=0)
            else:
                h_x2 = self.x_rnn(x)[-1]
            h_x = 0.5*h_x1 + 0.5*h_x2
        else:
            if self.x_birnn:
                h_x = self.x_rnn(x).mean(dim=0)
            else:
                h_x = self.x_rnn(x)[-1]

        return h_x

    def encode_y(self, y, fn):
        if self.gcn:
            y0 = y.permute(1, 2, 0).contiguous()
            h_y1 = self.x_gcn(y0)
            # h_y = h_y1
            h_y2 = self.e_rnn(y)
            h_y = 0.5*h_y1 + 0.5*h_y2
        else:
            h_y = self.e_rnn(y)
        h_y_ = h_y.clone()
        h_y = h_y.transpose(0, 1)
        h_y = h_y[fn == 1]

        return h_y, h_y_

    def encode(self, x, y, act, fn):
        h_x = self.encode_x(x)
        h_y, h_y_ = self.encode_y(y, fn)
        h_act = self.p_act_mlp(act)

        if self.text_label:
            if self.textencoder == "clip":
                labels = act.argmax(dim=1)
                h_text = self.text_label_feats[labels]
                h_text = self.text_mlp(h_text)
            elif self.textencoder == "roberta":
                labels = act.argmax(dim=1)
                h_text = self.text_label_feats[labels]
                h_text = torch.mean(self.textLinear(h_text), dim=1)
        # if self.text_label:
        #     # h_text = self.get_action_query(act)
        #     labels = act.argmax(dim=1).long()
        #
        #     if self.permute_text_label:
        #         text_labels = self.text_label_permutation[labels]
        #     else:
        #         text_labels = labels
        #
        #     if self.textencoder == "clip":
        #         h_text = self.text_label_feats[text_labels]
        #         h_text = self.text_mlp(h_text)
        #
        #     elif self.textencoder == "roberta":
        #         h_text = self.text_label_feats[text_labels]
        #         h_text = torch.mean(self.textLinear(h_text), dim=1)
        else:
            h_text = None

        h = torch.cat((h_x, h_y, h_act), dim=1)
        h = self.e_mlp(h)
        emu = self.e_mu(h)
        elogvar = self.e_logvar(h)

        h = torch.cat((h_x, h_act), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)

        return emu, elogvar, pmu, plogvar, h_x, h_act, h_y, h_y_, h_text

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    ## setting for sld-style ablation test
    # def get_anchor_features(self, batch_size, device, dtype):
    #     anchor_list = []
    #     for i in range(self.num_anchor):
    #         anchor_list.append(self.anchor_input[f'anchor_{i}'])
    #
    #     anchors = torch.cat(anchor_list, dim=0)  # [num_anchor, anchor_dim]
    #
    #     anchor_ids = torch.arange(batch_size, device=device) % self.num_anchor
    #     anchors = anchors[anchor_ids].to(device=device, dtype=dtype)
    #
    #     return anchors
    #
    # def get_sld_direction_from_hx(self, h_x, h_text=None, h_act=None):
    #     B = h_x.shape[0]
    #     anchors = self.get_anchor_features(B, h_x.device, h_x.dtype)
    #
    #     if h_text is not None:
    #         alpha_in = torch.cat([h_x, anchors, h_text], dim=1)
    #     else:
    #         alpha_in = torch.cat([h_x, anchors], dim=1)
    #
    #     alpha = self.sld_alpha_mlp(alpha_in)
    #     direction = self.direction(alpha)
    #     direction = self.dirLinearAfter(direction)
    #
    #     return direction, alpha

    def decode(self, x, z, h_act, h_x=None, h_dir=None, act=None, h_text=None):
        # dir_v = h_dir
        # h_dir = None

        z_save = z
        z = z_save

        if h_x is None:
            h_x = self.encode_x(x)
            h_x = self.skeleton2general(h_x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
            if self.text_label:
                self.d_rnn_text.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
            if self.text_label:
                self.d_rnn_text.initialize(batch_size=z.shape[0])
        y = []
        y_f = []
        h_act_ori = h_act.clone()
        if self.dir:
            # h_text = h_x
            dir = self.direction(self.dirLinear2(nn.functional.tanh(self.dirLinear1(h_text))))
            # dir_logits = self.dir_class(dir)
            dir = self.dirLinearAfter(dir)
            # h_act = h_act + self.dirLinearAfter(dir)

            # dir, alpha = self.get_sld_direction_from_hx(h_x, h_text=None, h_act=h_act)
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            if self.dir:
                if self.ddm:
                    if self.static_ddm:
                        h_act = h_act_ori + self.specs["decoder_fuse_w"] * self.ddm_w * dir
                    else:
                        h_act = h_act_ori + self.specs["decoder_fuse_w"] * torch.sigmoid(self.z_linear(z)+self.y_p_mlp(y_p))*dir
                else:
                    h_act = h_act_ori + self.specs["decoder_fuse_w"] * dir

            # 2026-04-28 attention failed
            # # -------- Q --------
            # q = self.W_q(torch.cat([z, y_p], dim=-1))  # [B, D]
            #
            # # -------- K, V --------
            # k = self.W_k(dir)  # [B, K, D]
            # v = self.W_v(dir)  # [B, K, D]
            #
            # # -------- Attention (einsum) --------
            # attn_score = torch.einsum('bd,bkd->bk', q, k)
            # attn = torch.softmax(attn_score / math.sqrt(self.attentionD), dim=-1)
            #
            # dir_attn = torch.einsum('bk,bkd->bd', attn, v)
            #
            # # -------- Fusion --------
            # h_act = h_act_ori + dir_attn

            # 2026-04-29 gate NoSum
            # -------- dynamic gate --------
            # feat = self.z_linear(z) + self.y_p_mlp(y_p)
            #
            # # D gate
            # gate_d = torch.sigmoid(feat)  # [B, D]
            # # K gate
            # alpha_k = torch.sigmoid(self.alpha_mlp(feat))  # [B, K]
            # alpha_k = alpha_k.unsqueeze(-1)  # [B, K, 1]
            #
            # dir_weighted = alpha_k * dir  # [B, K, D]
            # dir_agg = torch.sum(dir_weighted, dim=1)  # [B, D]
            #
            # h_act = h_act_ori + self.specs["decoder_fuse_w"] * dir_agg

            # h_act = h_act+h_text
            rnn_in = torch.cat([h_x, h_act, z, y_p], dim=1)
            h = self.d_rnn(rnn_in) # general space
            y_f.append(h)
            # h = self.general2skeleton(h) # skeleton space
            # h = torch.cat([h, dir_v[:,i]], dim=1)
            h = self.d_mlp(h)
            y_i = self.d_out(h) + x[-1]
            y.append(y_i)
        y = torch.stack(y)
        y_f = torch.stack(y_f)
        if self.use_gcn_decoder:
            y_f = y_f.permute(1, 2, 0).contiguous()
            gcn_out = self.y_gcn(y_f)
            final_out = 0.5*y + 0.5*gcn_out
        else:
            final_out = y
        return final_out, y_f

    def pred_act(self, y, fn, isfeats=False):
        if isfeats:
            h = y
        else:
            h, _ = self.encode_y(y, fn)
        # h = nn.functional.relu(self.pred_hid(h))
        # h = self.drop(h)
        logits = self.pred_head(h)
        # logits = self.classifier(y)

        return logits

    def latent_sim(self,y, fn, input_feat, ispred=False):
        fn = torch.argmax(fn, dim=-1)
        # _y = y[:fn]

        #
        # _y_motion = y[1:]-y[:-1]
        # # _y_motion = _y_motion.permute(1, 2, 0).contiguous()
        # # _y_motion = nn.functional.pad(_y_motion, (1, 0))
        # first = _y_motion[0:1, :, :]  # (1, b, v)，注意不要用 x[0]，否则会丢维度
        # # first = torch.zeros([1,_y_motion.shape[1],_y_motion.shape[2]]).cuda()
        # _y_motion = torch.cat([first, _y_motion], dim=0)
        # # last = _y_motion[-1:, :, :]
        # # last = torch.zeros([1, _y_motion.shape[1], _y_motion.shape[2]]).cuda()
        # # _y_motion = torch.cat([_y_motion, last], dim=0)

        #
        _y_motion = y

        # b, vc, t = _y_motion.shape
        # _y_motion = _y_motion.view(b, -1)
        _y_motion = self.latent_in(_y_motion)
        _y_feats = self.latent_mlp(_y_motion)
        # if ispred:
        #     input_feats = input_feat.view(b, -1)
        #     #input_feats = self.latent_mlp(input_feats)
        #     input_feats = self.latent_proj(input_feats)
        # else:
        #     input_feats = torch.cat(input_feat, dim=1)
        #     input_feats = self.latent_proj(input_feats)
        input_feats = input_feat

        return _y_feats, input_feats

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        timesteps: [T] or [bs] (int or float)
        return:    [T, dim]
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device="cuda") / half)
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x, y, act, fn, epoch=None):
        mu, logvar, pmu, plogvar, h_x, h_act, h_y, h_y_, h_text = self.encode(x, y, act, fn)
        z = self.reparameterize(mu, logvar) if self.training else mu
        # if self.dir:
        #     # act_f = self.dirLinear1(act)
        #     # q = self.vec_q(act_f).unsqueeze(1)
        #     # k = self.vec_k(act_f).unsqueeze(1)
        #     # v = self.input_v(input_b)
        #     # score = torch.einsum("bid,bjd->bij", q, k) / (self.vec_dim ** 0.5)
        #     # act_f = self.dir4hact(h_act)
        #
        #     ## no time dim
        #     act_f = self.dirLinear1(act).unsqueeze(1)
        #     act_f = torch.tanh(act_f)
        #     act_f = self.dirLinear2(act_f)
        #     h_dir = self.direction(act_f)
        #     h_dir = self.dirLinearAfter(h_dir).squeeze(1)
        #     pred, y_f = self.decode(x, z, h_act, h_x, h_dir)
        #
        #     ## use time dim
        #     # act_f = self.dirLinear1(act)
        #     # time = torch.arange(self.horizon, device="cuda")  # [horizon]
        #     # time_emb = self.timestep_embedding(time, 30)
        #     # time_f = self.time_mlp(time_emb)  # [T, 128]
        #     # act_f = act_f[:, None, :]  # [bs, 1, 128]
        #     # time_f = time_f[None, :, :]  # [1, T, 128]
        #     # dir_s = act_f + time_f
        #     # h_dir = self.direction(dir_s)
        #     # ## use rnn
        #     # # h_v = self.vecrnn_mlp(h_dir.mean(1))
        #     # # self.vec_rnn.initialize(batch_size=h_dir.shape[0], hx=h_v)
        #     # # vec_out = []
        #     # # for i in range(h_dir.shape[1]):
        #     # #     vec_pred = self.vec_rnn(h_dir[:, i])  # general space
        #     # #     vec_pred = self.vec_mlp(vec_pred)
        #     # #     vec_out.append(self.vec_out(vec_pred) + x[-1])
        #     # # vec_out = torch.stack(vec_out)
        #     # ## use_mlp
        #     # h_dir = h_dir.permute(0, 2, 1).contiguous()
        #     # h_dir = self.vec_tmlp(h_dir)
        #     # h_dir = h_dir.permute(0, 2, 1).contiguous()
        #     # vec_out = self.vec_mlp(h_dir).permute(1, 0, 2).contiguous()
        #     # pred, y_f = self.decode(x, z, h_act, h_x)
        #     # pred = 0.5*pred + 0.5*vec_out
        # else:
        #     pred, y_f = self.decode(x, z, h_act, h_x)
        pred, y_f = self.decode(x, z, h_act, h_x, act=act, h_text=h_text)
        if self.latent:
            # if self.text_label:
            #     gt_motion_f, latent_f = h_y, h_act
            #     # gt_motion_f, latent_f = self.latent_sim(y, fn, y_f, ispred=True)
            # else:
            #     # gt_motion_f, latent_f = self.latent_sim(y, fn, [h_x, h_act, z])
            #     gt_motion_f, latent_f = self.latent_sim(y, fn, y_f, ispred=True)
            #     # gt_motion_f, latent_f = h_y_, y_f
            h_y_l = h_y.detach()
            h_pred, _ = self.encode_y(pred, fn)
            gt_motion_f = h_y_l
            latent_f = h_pred
        else:
            gt_motion_f, latent_f = None, None
        if self.ar:
            ## old ar
            if epoch is not None and epoch < 999:
                y_logits = self.pred_act(h_y, fn, isfeats=True)
            else:
                y_logits = None
            pred_logits = self.pred_act(pred, fn)
            # # pred_logits = self.pred_act(h_act, fn, isfeats=True)
            return pred, mu, logvar, pmu, plogvar, y_logits, pred_logits, gt_motion_f, latent_f

            ## new ar
            # z_logits = self.pred_head_z(z)
            # h_logits = self.pred_head_h(y_f).mean(0)
            # return pred, mu, logvar, pmu, plogvar, z_logits, h_logits, gt_motion_f, latent_f

        else:
            return pred, mu, logvar, pmu, plogvar, gt_motion_f, latent_f

    def sample_prior(self, x, act):
        h_x = self.encode_x(x)
        # h_x = self.skeleton2general(h_x)
        h_act = self.p_act_mlp(act)
        if self.text_label:
            # h_text = self.get_action_query(act)
            if self.textencoder == "clip":
                labels = act.argmax(dim=1)
                h_text = self.text_label_feats[labels]
                h_text = self.text_mlp(h_text)
            elif self.textencoder == "roberta":
                labels = act.argmax(dim=1)
                h_text = self.text_label_feats[labels]
                h_text = torch.mean(self.textLinear(h_text), dim=1)
        # if self.text_label:
        #     # h_text = self.get_action_query(act)
        #     labels = act.argmax(dim=1).long()
        #
        #     if self.permute_text_label:
        #         text_labels = self.text_label_permutation[labels]
        #     else:
        #         text_labels = labels
        #
        #     if self.textencoder == "clip":
        #         h_text = self.text_label_feats[text_labels]
        #         h_text = self.text_mlp(h_text)
        #
        #     elif self.textencoder == "roberta":
        #         h_text = self.text_label_feats[text_labels]
        #         h_text = torch.mean(self.textLinear(h_text), dim=1)
        else:
            h_text = None

        h = torch.cat((h_x, h_act), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)
        z = self.reparameterize(pmu, plogvar)
        pred, y_f = self.decode(x, z, h_act, h_x, act=act, h_text=h_text)
        return pred


class Direction(nn.Module):
    def __init__(self, motion_dim, out_dim):
        super(Direction, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, input):
        # input: (bs*t) x out_dim

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:

            out = torch.einsum('bn,on->bo', input, Q)

            return out

class ActClassifier(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs):
        super(ActClassifier, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)

        # encode
        self.x_rnn = torch.nn.GRU(nx,nh_rnn,1)
        self.c_mlp = MLP(nh_rnn, nh_mlp, activation='relu', is_bn=specs.get('is_bn',False),
                         is_dropout=specs.get('is_dropout',False))
        self.c_out = nn.Linear(self.c_mlp.out_dim, self.n_action)

    def forward(self, x, fn,is_feat=False):
        h_x = self.x_rnn(x)[0] #[seq, bs, feat]
        h_x = h_x.transpose(0,1)
        h_x = h_x[fn==1] # [bs, feat]

        h_x = self.c_mlp(h_x)
        h = self.c_out(h_x)
        c = torch.softmax(h,dim=1)
        if is_feat:
            return c,h,h_x
        else:
            return c, h

def get_action_vae_model(cfg, traj_dim, model_version=None, max_len=None):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', None)

    # if model_name == 'v3_5_4':
    return MotionCompass(traj_dim, traj_dim, cfg.nz, max_len, specs, data=cfg.dataset, gcn=cfg.use_gcn, ar=cfg.use_ar, dir=cfg.use_dir, latent=cfg.use_latent, text_label=cfg.use_textlabel, ddm=cfg.use_ddm, actnames=cfg.train_actnames)


def get_action_classifier(cfg, traj_dim, model_version=None, max_len=None):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', None)
    return ActClassifier(traj_dim, traj_dim, cfg.nz, max_len, specs)

class gcn_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, data, joint,  **kwargs):
        super(gcn_encoder, self).__init__()
        # self.config = copy.deepcopy(config)

        skl = Skeleton(skl_type=data, joint_n=joint).skeleton
        skl = torch.tensor(skl, dtype=torch.float32, requires_grad=False)
        bi_skl = torch.zeros(joint, joint, requires_grad=False)
        bi_skl[skl != 0] = 1.
        self.A = bi_skl.cuda()
        self.A = self.A.unsqueeze(0)
        self.A.requires_grad = True
        in_channels = int(input_dim / joint)
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.joint = joint
        # kernel_size = 2
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, in_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(in_channels, in_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(in_channels, in_channels, kernel_size, 1, residual=False, **kwargs0),

            # st_gcn(in_channels, in_channels, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(64, 128, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(128, 256, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(128, 32, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(32, 4, kernel_size, 1, residual=False, **kwargs),
        ))
        self.L1 = nn.Linear(3 * self.joint, hidden_dim)
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size()))
            for i in self.st_gcn_networks
        ])

    def forward(self, motion_feats):
        b, vc, t = motion_feats.shape
        x = self.data_bn(motion_feats)
        x = x.reshape(b, self.joint, 3, t)
        x = x.permute(0, 2, 3, 1).contiguous()  # bs, c, t, v
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        skeleton_feats = x.permute(0, 3, 1, 2).contiguous().reshape(b, -1, t).transpose(1, 2)
        skeleton_feats = self.L1(skeleton_feats).transpose(0, 1)
        # skeleton_feats = skeleton_feats.transpose(0, 1)

        return skeleton_feats

class gcn_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, data, joint, **kwargs):
        super(gcn_decoder, self).__init__()
        # self.config = copy.deepcopy(config)

        skl = Skeleton(skl_type=data, joint_n=joint).skeleton
        skl = torch.tensor(skl, dtype=torch.float32, requires_grad=False)
        bi_skl = torch.zeros(joint, joint, requires_grad=False)
        bi_skl[skl != 0] = 1.
        self.A = bi_skl.cuda()
        self.A = self.A.unsqueeze(0)
        self.A.requires_grad = True
        # in_channels = int(input_dim / joint)
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(input_dim)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.joint = joint
        # kernel_size = 2
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(3, 3, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(3, 3, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(3, 3, kernel_size, 1, residual=False, **kwargs0),

            # st_gcn(in_channels, in_channels, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(64, 128, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(128, 256, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(128, 32, kernel_size, 1, residual=False, **kwargs),
            # st_gcn(32, 4, kernel_size, 1, residual=False, **kwargs),
        ))
        self.L2 = nn.Linear(input_dim, 3 * self.joint)
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size()))
            for i in self.st_gcn_networks
        ])

    def forward(self, motion_feats):
        b, vc, t = motion_feats.shape
        x = self.data_bn(motion_feats)
        x = self.L2(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = x.reshape(b, self.joint, 3, t)
        x = x.permute(0, 2, 3, 1).contiguous()  # bs, c, t, v
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        skeleton_feats = x.permute(0, 3, 1, 2).contiguous().reshape(b, -1, t).permute(2, 0, 1).contiguous()
        # skeleton_feats = self.L1(skeleton_feats).transpose(0, 1)
        # skeleton_feats = skeleton_feats.transpose(0, 1)

        return skeleton_feats