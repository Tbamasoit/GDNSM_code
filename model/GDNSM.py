import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.utils import init_seed
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start



def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_to_neg_one_to_one(emb):
    return emb * 2 - 1


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Encoder_CFG(nn.Module):
    def __init__(self, config, in_ft, out_ft, v_dim, t_dim) -> None:
        super(Encoder_CFG, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(out_ft),
            nn.Linear(out_ft, out_ft*2),
            nn.GELU(),
            nn.Linear(out_ft*2, out_ft),
        )
        self.out_ft = out_ft
        self.v_dim = v_dim
        self.t_dim = t_dim
        self.v_mlp = nn.Linear(self.out_ft, out_ft)
        self.t_mlp = nn.Linear(self.out_ft, out_ft)
        if config['dataset'] == 'baby' or config['dataset'] == 'tiktok': 
            self.diffuser = nn.Sequential(nn.Linear(self.out_ft*5, self.out_ft))
        else:
            self.diffuser = nn.Sequential(       
                nn.Linear(self.out_ft*5, self.out_ft*2),
                nn.GELU(),
                nn.Linear(self.out_ft*2, self.out_ft)
            )
        self.v_none_embedding = nn.Embedding(  
            num_embeddings=1,
            embedding_dim=self.out_ft,
        )
        self.t_none_embedding = nn.Embedding(  
            num_embeddings=1,
            embedding_dim=self.out_ft,
        )
        

    def forward_v(self, x, t, y):
        device = x.device
        t = self.time_mlp(t)
        u_cond, t_cond, v_cond = torch.split(y, [self.out_ft, self.out_ft, self.out_ft], dim=1)
        h = self.t_none_embedding(torch.tensor([0]).to(device))
        h = torch.cat([h.view(1, self.out_ft)]*x.shape[0], dim=0) 
        v_cond = self.v_mlp(v_cond)
        res = self.diffuser(torch.cat((x, t, u_cond, h, v_cond), dim=1))
        return res
    
    def forward_t(self, x, t, y):
        device = x.device
        t = self.time_mlp(t)
        u_cond, t_cond, v_cond = torch.split(y, [self.out_ft, self.out_ft, self.out_ft], dim=1)
        h = self.v_none_embedding(torch.tensor([0]).to(device))
        h = torch.cat([h.view(1, self.out_ft)]*x.shape[0], dim=0)
        t_cond = self.t_mlp(t_cond)
        res = self.diffuser(torch.cat((x, t, u_cond, t_cond, h), dim=1))
        return res

    def forward_vt(self, x, t, y):
        device = x.device
        t = self.time_mlp(t)
        u_cond, t_cond, v_cond = torch.split(y, [self.out_ft, self.out_ft, self.out_ft], dim=1)
        t_cond = self.t_mlp(t_cond)
        v_cond = self.v_mlp(v_cond)
        res = self.diffuser(torch.cat((x, t, u_cond, t_cond, v_cond), dim=1))
        return res
    
    def forward_uncon(self, x, t, y):
        device = x.device
        t = self.time_mlp(t)
        u_cond, t_cond, v_cond = torch.split(y, [self.out_ft, self.out_ft, self.out_ft], dim=1)
        h_t = self.t_none_embedding(torch.tensor([0]).to(device))
        h_v = self.v_none_embedding(torch.tensor([0]).to(device))
        h_t = torch.cat([h_t.view(1, self.out_ft)]*x.shape[0], dim=0) 
        h_v = torch.cat([h_v.view(1, self.out_ft)]*x.shape[0], dim=0)
        res = self.diffuser(torch.cat((x, t, u_cond, h_t, h_v), dim=1))
        return res

    def cacu_cond(self, y, p):
        device = y.device
        u_cond, t_cond, v_cond = torch.split(y, [self.out_ft, self.out_ft, self.out_ft], dim=1)
        B, D = y.shape[0], y.shape[1]
        mask1d = (torch.sign(torch.rand(B) - 3*p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(device)
        non_t = self.t_none_embedding(torch.tensor([0]).to(device))
        non_v = self.v_none_embedding(torch.tensor([0]).to(device))
        non_t = torch.cat([non_t.view(1, self.out_ft)]*B, dim=0) 
        non_v = torch.cat([non_v.view(1, self.out_ft)]*B, dim=0)
        no_tv = torch.cat([u_cond, non_t, non_v], dim=1)
        no_t = torch.cat([u_cond, non_t, v_cond], dim=1)
        no_v = torch.cat([u_cond, t_cond, non_t], dim=1)
        y = y * mask + no_tv * (1-mask)/3 + no_t * (1-mask)/3 + no_v * (1-mask)/3
        return y


class Diffusion_CFG(nn.Module):
    def __init__(self, config, in_feat, out_feat, timesteps, v_dim, t_dim) -> None:
        super(Diffusion_CFG, self).__init__()

        self.encoder = Encoder_CFG(config, in_feat, out_feat, v_dim, t_dim)

        self.timesteps = timesteps 

        # define beta schedule
        self.loss_type = config['loss_type']
        if self.loss_type == None:
            raise NotImplementedError()
        if config['scheduler'] == 'cosine_beta_schedule':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif config['scheduler'] == 'linear_beta_schedule':
            self.betas = linear_beta_schedule(timesteps=self.timesteps)
        elif config['scheduler'] == 'quadratic_beta_schedule':
            self.betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif config['scheduler'] == 'sigmoid_beta_schedule':
            self.betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            raise NotImplementedError()
        
        self.v_cfg_scale = config['cfg_scale'] 
        self.t_cfg_scale = config['cfg_scale']
        self.vt_cfg_scale = config['cfg_scale']
        self.p = 0.05

        # define alphas 
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        self.alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    
    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def p_losses(self, x_start, t, labels, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    
        predicted_noise = self.encoder.forward_vt(x_noisy, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def p_sample(self, model, x, t, labels, t_index, flag): 
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        uncond_predicted_noise = model.forward_uncon(x, t, labels)
        v_predicted_noise = model.forward_v(x, t, labels)
        t_predicted_noise = model.forward_t(x, t, labels)
        predicted_noise = model.forward_vt(x, t, labels)
        if flag == 1: # v_guided
            predicted_noise = uncond_predicted_noise + self.v_cfg_scale*(v_predicted_noise-uncond_predicted_noise)
        # t_diffusion
        elif flag == 2: # t_guided
            predicted_noise = uncond_predicted_noise + self.t_cfg_scale*(t_predicted_noise-uncond_predicted_noise)
        # vt_diffusion
        elif flag == 3: # vt_guided
            predicted_noise = uncond_predicted_noise + self.v_cfg_scale*(v_predicted_noise-uncond_predicted_noise) + self.t_cfg_scale*(predicted_noise-v_predicted_noise)
        # predicted_noise = uncond_predicted_noise + self.vt_cfg_scale*(predicted_noise-uncond_predicted_noise) 
        else:
            predicted_noise = uncond_predicted_noise
            
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    

    def p_sample_loop(self, model, shape, y, flag):
        device = next(model.parameters()).device

        b = shape[0]
        emb = torch.randn(shape, device=device)
        embs = []

        for i in reversed(range(0, self.timesteps)):
            emb = self.p_sample(model, emb, torch.full((b,), i, device=device, dtype=torch.long), y, i, flag)
            embs.append(emb)

        embs = embs[::-1]
        steps = [0, int(self.timesteps/10), int(self.timesteps/8), int(self.timesteps/4), int(self.timesteps/2)]
        
        out = [embs[step] for step in steps]
        return out


    @torch.no_grad()
    def sample(self, shape, y, flag):
        return self.p_sample_loop(self.encoder, shape, y, flag)


    def forward(self, input, labels, device): 
        t = torch.randint(0, self.timesteps, (input.shape[0],), device=device).long()
        labels = self.encoder.cacu_cond(labels, self.p)
        return self.p_losses(input, t, labels, loss_type=self.loss_type)


class GDNSM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GDNSM, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k'] # 10
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        
        self.timesteps = config['timesteps']
        self.d_epoch = config['d_epoch']
        self.no_diff = True
        self.mm_diff = False
        self.diffusion_MM = Diffusion_CFG(config, self.embedding_dim, self.embedding_dim, self.timesteps, self.v_feat.shape[1], self.t_feat.shape[1])

        # load dataset info  TrainDataLoader.inter_matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32) # .shape [n_users, n_items]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight) 

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        self.norm_adj = self.get_adj_mat() # [n_users+n_items, n_users+n_items]
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device) 
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None: 
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym') # knn weighted each element before norm is float not {0, 1}
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)


    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self): # [n_users, n_items]->[n_users+n_items, n_users+n_items]
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight) # mlp
        if self.t_feat is not None: 
            text_feats = self.text_trs(self.text_embedding.weight) # mlp

        image_item_embeds = self.item_id_embedding.weight 
        text_item_embeds = self.item_id_embedding.weight

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        side_embeds = (0.1*image_embeds + 0.9*text_embeds)/3 # ab
        
        all_embeds = content_embeds + side_embeds # content->id side->MM

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0] # [training_batch_size]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0] # [eval_batch_size]
        
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
#目的：责任分离” (Separation of Concerns
#让GDNSM模型定义神经网络架构和计算逻辑（前向传播、损失计算），计算核心
#MyDataset类处理所有与数据相关的“脏活累活”，数据工程师
    
#请把我已经计算好的邻接矩阵和特征张量直接递给我，我立刻就能开始工作
#希望它的 __init__ 签名变成 (self, config, adj_matrix, features)  
class MyGDNSM(GeneralRecommender):
    def __init__(self, config, dataset_info):
        super().__init__(config)#不再需要传入dataset
        #所有config传入参数
        self.sparse = True
        self.device = config['device']
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k'] # 10
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        
        self.timesteps = config['timesteps']
        self.d_epoch = config['d_epoch']
        self.no_diff = True
        self.mm_diff = False

        #所有从dataset_info中传入的参数
        self.n_users = dataset_info['n_users']
        self.n_items = dataset_info['n_items']
        self.norm_adj = dataset_info['norm_adj'].to(self.device)
        self.R = dataset_info['R_tensor'].to(self.device)

        # 多模态图也需要移动，否则后面还会报错
        if dataset_info['image_adj'] is not None:
            self.image_original_adj = dataset_info['image_adj'].to(self.device)
        
        if dataset_info['text_adj'] is not None:
            self.text_original_adj = dataset_info['text_adj'].to(self.device)

        # 特征矩阵通常用于初始化 Embedding，Embedding 层会自动处理设备移动
        # 但为了保险起见，如果后面有直接使用 feats 的地方，也可以移一下
        self.v_feat = dataset_info['v_feat'].to(self.device) if dataset_info['v_feat'] is not None else None
        self.t_feat = dataset_info['t_feat'].to(self.device) if dataset_info['t_feat'] is not None else None

        self.diffusion_MM = Diffusion_CFG(config, self.embedding_dim, self.embedding_dim, self.timesteps, self.v_feat.shape[1], self.t_feat.shape[1])

        # load dataset info  TrainDataLoader.inter_matrix
        # self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32) # .shape [n_users, n_items]

        #初始化embedding层
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        # image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        # text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        #self.norm_adj = self.get_adj_mat() # [n_users+n_items, n_users+n_items]
        #self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device) 
        #self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # if self.v_feat is not None: 
        #     self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        #     if os.path.exists(image_adj_file):
        #         image_adj = torch.load(image_adj_file)
        #     else:
        #         image_adj = build_sim(self.image_embedding.weight.detach())
        #         image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
        #                                                norm_type='sym') # knn weighted each element before norm is float not {0, 1}
        #         torch.save(image_adj, image_adj_file)
        #     self.image_original_adj = image_adj.cuda()

        # if self.t_feat is not None:
        #     self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        #     if os.path.exists(text_adj_file):
        #         text_adj = torch.load(text_adj_file)
        #     else:
        #         text_adj = build_sim(self.text_embedding.weight.detach())
        #         text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        #         torch.save(text_adj, text_adj_file)
        #     self.text_original_adj = text_adj.cuda()

        # if self.v_feat is not None:
        #     self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        # if self.t_feat is not None:
        #     self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)


    def pre_epoch_processing(self):
        pass



    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight) # mlp
        if self.t_feat is not None: 
            text_feats = self.text_trs(self.text_embedding.weight) # mlp

        image_item_embeds = self.item_id_embedding.weight 
        text_item_embeds = self.item_id_embedding.weight

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        side_embeds = (0.1*image_embeds + 0.9*text_embeds)/3 # ab
        
        all_embeds = content_embeds + side_embeds # content->id side->MM

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    # 修改签名，增加 generated_negs 参数
    def calculate_loss(self, interaction, generated_negs=None):
        users = interaction[0] # [training_batch_size]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        # 1. 计算标准的 BPR 和 CL Loss
        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        recommender_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss
        
        # 2. [新增] 如果有生成的困难负样本，计算额外的 BPR Loss
        #  [适配重构] 处理生成的困难负样本
        if generated_negs is not None:
            # 注意：generated_negs 是嵌入向量，不是 ID
            # generated_negs shape: [Batch, g(epoch), Dim]
            # u_g_embeddings shape: [Batch, Dim] -> 需要 unsqueeze 变成 [Batch, 1, Dim]
            
            user_emb_expanded = u_g_embeddings.unsqueeze(1) # [B, 1, D]
            
            # 我们需要计算它和用户嵌入的得分
            # 计算点积分数: [B, 1, D] * [B, g, D] -> sum(dim=-1) -> [B, g]
            gen_neg_scores = (user_emb_expanded * generated_negs).sum(dim=-1)

            # 正样本分数扩展: pos_scores [B] -> [B, 1]
            pos_scores_expanded = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=1, keepdim=True)

            
            # generated_neg_scores = torch.sum(u_g_embeddings * generated_negs, dim=1)
            # pos_scores = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=1)
            
            # 论文中的 L_NEG (公式 24)
            # 计算 Loss (广播机制: [B, 1] - [B, g] -> [B, g])
            # 对 g 个负样本的 Loss 求平均
            gen_loss = -torch.mean(F.logsigmoid(pos_scores_expanded - gen_neg_scores))            
            # 从 config 中获取权重
            recommender_loss += self.config['lambda_neg']* gen_loss

        return recommender_loss

    def full_sort_predict(self, interaction):
        user = interaction[0] # [eval_batch_size]
        
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
    # in models/GDNSM.py -> class GDNSM

    def get_diffusion_inputs(self, users, pos_items):
        """
        为扩散模型的训练和采样准备输入。
        返回：正样本嵌入，以及拼接好的条件信息 (user, text, visual)。
        """
        # 1. 获取所有嵌入
        ua_embeddings, ia_embeddings, _, _ = self.forward(self.norm_adj, train=True)
        
        # 2. 提取当前批次的正样本嵌入
        pos_item_embeds = ia_embeddings[pos_items]
        user_embeds = ua_embeddings[users]
        
        # 3. 提取对应的多模态特征并转换维度
        # 注意：这里需要 self.text_trs 和 self.image_trs 两个 MLP
        pos_text_feats = self.text_trs(self.text_embedding(pos_items))
        pos_visual_feats = self.image_trs(self.image_embedding(pos_items))
        
        # 4. 拼接成 Diffusion_CFG.forward 所需的 'labels' 格式
        diffusion_conditions = torch.cat([user_embeds, pos_text_feats, pos_visual_feats], dim=1)
        
        return pos_item_embeds, diffusion_conditions
    

    @torch.no_grad()
    def generate_batch_negatives(self, users, pos_items):
        """
        [新增接口] 为当前 Batch 的用户生成三个难度级别的负样本。
        供 Trainer 调用。
        
        Returns:
            neg_v, neg_t, neg_vt: 每一个形状都是 [Batch_size, M, Dim]
        """
        # 1. 准备条件 (重用你已经写的 get_diffusion_inputs)
        # 注意：这里我们只需要 conditions，不需要 pos_item_embeds
        _, diffusion_conditions = self.get_diffusion_inputs(users, pos_items)
        
        batch_size = users.shape[0]
        # 假设生成样本数 M 从 config 获取，或者默认为 1
        # 这里的 M 需要与 Scheduler 中的 M 保持一致
        # 我们假设 Diffusion_CFG.sample 支持生成 M 个样本 (需要确认 sample 实现)
        # 如果 sample 只生成 1 个，形状是 [Batch, Dim]，我们需要 reshape 为 [Batch, 1, Dim]
        
        shape = (batch_size, self.embedding_dim) 
        
        # 2. 调用扩散模型生成
        # flag=1: Visually similar (Easy)
        neg_v_list = self.diffusion_MM.sample(shape, diffusion_conditions, flag=1)
        # sample 返回的是 list of tensors (time steps)，我们通常取最后一步 [-1]
        neg_v = neg_v_list[-1].unsqueeze(1) # [Batch, 1, Dim] (假设 M=1)

        # flag=2: Textually similar (Medium)
        neg_t_list = self.diffusion_MM.sample(shape, diffusion_conditions, flag=2)
        neg_t = neg_t_list[-1].unsqueeze(1)

        # flag=3: Both similar (Hard)
        neg_vt_list = self.diffusion_MM.sample(shape, diffusion_conditions, flag=3)
        neg_vt = neg_vt_list[-1].unsqueeze(1)
        
        return neg_v, neg_t, neg_vt




