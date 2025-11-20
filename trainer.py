import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from time import time
from logging import getLogger
from GDNSM.utils.topk_evaluator import TopKEvaluator
from GDNSM.utils.utils import early_stopping, dict2str
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np



class Trainer:
    def __init__(self, config, model, diffusion_model=None, mg=False):
        """
        GDNSM Trainer，带 diffusion 生成负样本
        """
        self.config = config
        self.model = model
        self.diffusion_model = diffusion_model  # 生成负样本的模型
        self.mg = mg
        self.logger = getLogger()

        # 配置参数
        self.learner = getattr(config, 'learner', 'adam')
        self.learning_rate = getattr(config, 'learning_rate', 0.001)
        self.epochs = getattr(config, 'epochs', 100)
        self.eval_step = min(getattr(config, 'eval_step', 1), self.epochs)
        self.stopping_step = getattr(config, 'stopping_step', 10)
        self.clip_grad_norm = getattr(config, 'clip_grad_norm', None)
        self.valid_metric = getattr(config, 'valid_metric', 'NDCG@20').lower()
        self.valid_metric_bigger = getattr(config, 'valid_metric_bigger', True)
        self.test_batch_size = getattr(config, 'eval_batch_size', 128)
        self.device = getattr(config, 'device', 'cpu')
        self.weight_decay = getattr(config, 'weight_decay', 0.0)
        if isinstance(self.weight_decay, str):
            self.weight_decay = eval(self.weight_decay)

        self.req_training = getattr(config, 'req_training', True)
        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {f'{j.lower()}@{k}': 0.0 for j, k in itertools.product(
            getattr(config, 'metrics', ['NDCG']), getattr(config, 'topk', [20])
        )}
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        # self.optimizer = self._build_optimizer()

        # lr_scheduler = getattr(config, 'learning_rate_scheduler', [0.96, 50])
        # fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        # self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        # === 新增两个优化器，分别更新 φ 和 θ ===
        theta_params = list(self.model.diffusion_MM.parameters())
        theta_param_ids = set(id(p) for p in theta_params)
        phi_params = [p for p in self.model.parameters() if id(p) not in theta_param_ids]

        self.opt_phi = optim.Adam(phi_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.opt_theta = optim.Adam(theta_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # === 各自配一个 scheduler ===
        lr_scheduler = getattr(config, 'learning_rate_scheduler', [0.96, 50])
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])

        self.lr_phi = optim.lr_scheduler.LambdaLR(self.opt_phi, lr_lambda=fac)
        self.lr_theta = optim.lr_scheduler.LambdaLR(self.opt_theta, lr_lambda=fac)


        self.eval_type = getattr(config, 'eval_type', 'full')
        self.evaluator = TopKEvaluator(config)

        # GDNSM 多目标训练参数
        self.alpha1 = getattr(config, 'alpha1', 1.0)
        self.alpha2 = getattr(config, 'alpha2', 1.0)
        self.beta = getattr(config, 'beta', 1)

    def _build_optimizer(self):
        opt_dict = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop
        }
        OptimCls = opt_dict.get(self.learner.lower(), optim.Adam)
        if OptimCls == optim.Adam and self.learner.lower() not in opt_dict:
            self.logger.warning('Unrecognized optimizer, using default Adam')
        return OptimCls(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False):
        self.model.eval()
        batch_matrix_list = []
        for batch in eval_data:
            scores = self.model.full_sort_predict(batch)
            masked_items = batch[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(getattr(self.config, 'topk', [20])), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test)

    # def _train_epoch_bprcl_diffu(self, train_data):
    #     self.model.train()
    #     total_loss = 0.0

    #     # 配置参数
    #     d_epoch = getattr(self.config, 'd_epoch', 1)
    #     use_mm_diff = getattr(self.config, 'use_mm_diff', True)
    #     base_cl_weight = getattr(self.config, 'cl_weight', 0.01)
    #     max_cl_weight = getattr(self.config, 'max_cl_weight', 0.1)
    #     beta = getattr(self.config, 'beta', 1.0)
    #     epoch = getattr(self, 'current_epoch', 0)
    #     E = max(getattr(self.config, 'epochs', 100), 1)
    #     cl_weight = base_cl_weight + (max_cl_weight - base_cl_weight) * (epoch / E)

    #     for batch in train_data:
    #         users = batch[0]
    #         pos_items = batch[1]

    #         if len(batch) >= 3:
    #             neg_items = batch[2]
    #         else:
    #             num_items = self.model.n_items
    #             neg_items = torch.randint(0, num_items, pos_items.shape, device=pos_items.device)

    #         # ===== Phase A: Diffusion θ =====
    #         if use_mm_diff:
    #             for p in self.model.parameters():
    #                 p.requires_grad = False
    #             for p in self.model.diffusion_MM.parameters():
    #                 p.requires_grad = True

    #             for _ in range(d_epoch):
    #                 with torch.no_grad():
    #                     ua_embeddings, ia_embeddings, _, _ = self.model.forward(self.model.norm_adj, train=True)
    #                     u_g = ua_embeddings[users]
    #                     x0 = ia_embeddings[pos_items]

    #                     t_cond = self.model.text_trs(self.model.text_feat[pos_items])
    #                     v_cond = self.model.image_trs(self.model.image_feat[pos_items])
    #                     labels = torch.cat([u_g, t_cond, v_cond], dim=1)

    #                 diff_loss = self.model.diffusion_MM(x0, labels, device=x0.device)

    #                 self.opt_theta.zero_grad()
    #                 diff_loss.backward()
    #                 self.opt_theta.step()

    #             for p in self.model.parameters():
    #                 p.requires_grad = True
    #             for p in self.model.diffusion_MM.parameters():
    #                 p.requires_grad = False

    #         # ===== Phase B: φ (BPR + CL + Diffusion Negative) =====
    #         ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
    #         u_g = ua_embeddings[users]
    #         pos_g = ia_embeddings[pos_items]
    #         rand_neg_g = ia_embeddings[neg_items]

    #         # -------- Contrastive Learning Loss --------
    #         side_u, side_i = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
    #         cont_u, cont_i = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
    #         side_u, side_i = F.normalize(side_u, dim=1), F.normalize(side_i, dim=1)
    #         cont_u, cont_i = F.normalize(cont_u, dim=1), F.normalize(cont_i, dim=1)
    #         cl_loss = self.model.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
    #                 self.model.InfoNCE(side_u[users], cont_u[users], 0.2)

    #         # -------- Diffusion生成负样本并应用动态难度调度 --------
    #         L_NEG = torch.tensor(0.0, device=pos_g.device)
    #         if use_mm_diff:
    #             neg_sets = []
    #             with torch.no_grad():
    #                 t_cond = self.model.text_trs(self.model.text_feat[pos_items])
    #                 v_cond = self.model.image_trs(self.model.image_feat[pos_items])
    #                 labels_gen = torch.cat([u_g, t_cond, v_cond], dim=1)

    #                 # 生成三种难度的负样本 (Difficulty level 1-1, 1-2, 2)
    #                 for flag in (1, 2, 3):
    #                     steps = self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=flag)
    #                     neg_sets.append(steps[-1])  # 取最后一步作为负样本

    #             all_negs = torch.stack(neg_sets, dim=1)  # shape: [batch, difficulty, dim]
    #             u_norm = F.normalize(u_g, dim=1).unsqueeze(1)
    #             n_norm = F.normalize(all_negs, dim=2)
    #             cos = (u_norm * n_norm).sum(dim=2)
    #             _, order = torch.sort(cos, dim=1, descending=False)  # 从最相似(困难)到最不相似(简单)

    #             # 动态难度调度 (Curriculum Learning)
    #             Mmax = 3
    #             g = max(1, min(Mmax, int(round(Mmax * (epoch+1) / E))))  # 随 epoch 增加选择更多难样本
    #             for k in range(g):
    #                 idx = order[:, k]
    #                 pick = all_negs[torch.arange(all_negs.size(0)), idx]
    #                 mf, _, _ = self.model.bpr_loss(u_g, pos_g, pick)
    #                 L_NEG += mf
    #             L_NEG /= g

    #         # -------- 原始 BPR loss --------
    #         bpr_mf, bpr_emb, bpr_reg = self.model.bpr_loss(u_g, pos_g, rand_neg_g)

    #         # -------- 总 loss --------
    #         total = bpr_mf + bpr_emb + bpr_reg + cl_weight * cl_loss
    #         if use_mm_diff:
    #             total += beta * L_NEG

    #         self.opt_phi.zero_grad()
    #         total.backward()
    #         self.opt_phi.step()

    #         total_loss += total.item()

    #     self.current_epoch = epoch + 1
    #     return total_loss / len(train_data)
    def _train_epoch_phi(self, train_data, epoch):
        self.model.train()
        total_loss = 0.0
        for batch in train_data:
            users, pos_items = batch[0], batch[1]
            neg_items = torch.randint(0, self.model.n_items, pos_items.shape, device=pos_items.device)

            ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
            u_g, pos_g, neg_g = ua_embeddings[users], ia_embeddings[pos_items], ia_embeddings[neg_items]

            # BPR
            bpr_mf, bpr_emb, bpr_reg = self.model.bpr_loss(u_g, pos_g, neg_g)

            # CL
            side_u, side_i = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cont_u, cont_i = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cl_loss = self.model.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
                    self.model.InfoNCE(side_u[users], cont_u[users], 0.2)

            total = bpr_mf + bpr_emb + bpr_reg +  0.01 *cl_loss

            self.opt_phi.zero_grad()
            total.backward()
            self.opt_phi.step()
            total_loss += total.item()

        return total_loss / len(train_data)

    def _train_epoch_theta(self, train_data, epoch):
        self.model.train()
        total_loss = 0.0

        # φ 冻结
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.diffusion_MM.parameters():
            p.requires_grad = True

        for batch in train_data:
            users, pos_items = batch[0], batch[1]

            with torch.no_grad():
                ua_embeddings, ia_embeddings, _, _ = self.model.forward(self.model.norm_adj, train=True)
                u_g = ua_embeddings[users]
                x0 = ia_embeddings[pos_items]
                t_cond = self.model.text_trs(self.model.text_feat[pos_items])
                v_cond = self.model.image_trs(self.model.image_feat[pos_items])
                labels = torch.cat([u_g, t_cond, v_cond], dim=1)

            diff_loss = self.model.diffusion_MM(x0, labels, device=x0.device)
            self.opt_theta.zero_grad()
            diff_loss.backward()
            self.opt_theta.step()
            total_loss += diff_loss.item()

        # 解冻回来
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.diffusion_MM.parameters():
            p.requires_grad = False

        return total_loss / len(train_data)

    def _train_epoch_joint(self, train_data, epoch):
        self.model.train()
        total_loss = 0.0

        # 用于保存相似度
        cos_sim_pos_list = []
        cos_sim_vneg_list = []
        cos_sim_tneg_list = []
        cos_sim_tvneg_list = []

        # 动态难度调度参数
        S = getattr(self.config, 'sched_S', 30)        # pacing 参数 S
        lam = getattr(self.config, 'sched_lambda', 1)  # pacing 参数 λ
        M = getattr(self.config, 'num_M', 5)           # 基础负样本数 (总共 3M)

        def g(ep):
            if ep < S:
                return 0
            else:
                return int(min(3 * M, ((ep / S) ** lam) * M))

        for batch in train_data:
            users, pos_items = batch[0], batch[1]

            # 正样本编码
            ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
            u_g, pos_g = ua_embeddings[users], ia_embeddings[pos_items]

            # 随机负样本用于 BPR
            neg_items = torch.randint(0, self.model.n_items, (pos_items.shape[0],), device=pos_items.device)
            neg_g = ia_embeddings[neg_items]
            bpr_mf, bpr_emb, bpr_reg = self.model.bpr_loss(u_g, pos_g, neg_g)

            # CL
            side_u, side_i = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cont_u, cont_i = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cl_loss = self.model.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
                    self.model.InfoNCE(side_u[users], cont_u[users], 0.2)

            # θ 生成负样本 (固定 θ)
            with torch.no_grad():
                t_cond = self.model.text_trs(self.model.text_feat[pos_items])
                v_cond = self.model.image_trs(self.model.image_feat[pos_items])
                labels_gen = torch.cat([u_g, t_cond, v_cond], dim=1)

                # 三种生成负样本
                O_v_all, O_t_all, O_tv_all = [], [], []
                for _ in range(M):
                    O_v_all.append(self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=1)[-1])
                    O_t_all.append(self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=2)[-1])
                    O_tv_all.append(self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=3)[-1])


            with torch.no_grad(): 
                # 计算余弦相似度
                u_norm = F.normalize(u_g, dim=1)             # [B, D]
                pos_norm = F.normalize(pos_g, dim=1)
                for vneg, tneg, tvneg in zip(O_v_all, O_t_all, O_tv_all):
                    vneg_norm = F.normalize(vneg, dim=1)
                    tneg_norm = F.normalize(tneg, dim=1)
                    tvneg_norm = F.normalize(tvneg, dim=1)
                    cos_sim_pos_list.extend((u_norm * pos_norm).sum(dim=1).cpu().numpy())
                    cos_sim_vneg_list.extend((u_norm * vneg_norm).sum(dim=1).cpu().numpy())
                    cos_sim_tneg_list.extend((u_norm * tneg_norm).sum(dim=1).cpu().numpy())
                    cos_sim_tvneg_list.extend((u_norm * tvneg_norm).sum(dim=1).cpu().numpy())

            # 动态难度调度
            Q_diff = O_v_all + O_t_all + O_tv_all
            Q_diff = Q_diff[:g(epoch+1)]


            # 计算 L_NEG
            L_NEG = 0.0
            for hard_neg in Q_diff:
                mf_neg, _, _ = self.model.bpr_loss(u_g, pos_g, hard_neg)
                L_NEG += mf_neg
            if len(Q_diff) > 0:
                L_NEG /= len(Q_diff)
            if epoch<S:
                L_NEG = 0.0

            # 总损失
            total = bpr_mf + bpr_emb + bpr_reg + 0.01* cl_loss + L_NEG
            self.opt_phi.zero_grad()
            total.backward()
            self.opt_phi.step()
            total_loss += total.item()

        # 打印余弦相似度分布
        print("Cosine similarity distributions:")
        print(f"Positive samples: mean={np.mean(cos_sim_pos_list):.4f}, std={np.std(cos_sim_pos_list):.4f}")
        print(f"Video Negatives: mean={np.mean(cos_sim_vneg_list):.4f}, std={np.std(cos_sim_vneg_list):.4f}")
        print(f"Text Negatives: mean={np.mean(cos_sim_tneg_list):.4f}, std={np.std(cos_sim_tneg_list):.4f}")
        print(f"Video+Text Negatives: mean={np.mean(cos_sim_tvneg_list):.4f}, std={np.std(cos_sim_tvneg_list):.4f}")

        return total_loss / len(train_data)



    # def fit(self, train_data, valid_data=None, test_data=None, saved=True, verbose=True):
    #     """
    #     三阶段训练:
    #     1. 推荐模型 φ 预训练 (BPR+CL)
    #     2. 扩散模型 θ 训练 (φ 固定)
    #     3. 联合阶段 (θ 固定, φ 更新, 生成负样本)
    #     """
    #     E1 = getattr(self.config, 'pretrain_epochs', 10)
    #     E2 = getattr(self.config, 'diff_epochs', 10)
    #     E3 = getattr(self.config, 'joint_epochs', 30)

    #     self.best_valid_score = -1e9 if self.valid_metric_bigger else 1e9
    #     self.best_valid_result, self.best_test_upon_valid = {}, {}

    #     print(f"\n===> 开始三阶段训练，总轮数 {E1+E2+E3}")

    #     # -------- 阶段 1: 推荐模型 φ --------
    #     print(f"\n[阶段 1] 预训练推荐模型 φ ({E1} 轮)")
    #     for epoch in range(E1):
    #         loss = self._train_epoch_phi(train_data, epoch)
    #         if verbose:
    #             self.logger.info(f"[阶段 1] Epoch {epoch+1}/{E1} | Loss={loss:.4f}")
    #         # 验证
    #         if valid_data and (epoch + 1) % self.eval_step == 0:
    #             self._do_validation(epoch, valid_data, test_data, saved, verbose)

    #     # -------- 阶段 2: 扩散模型 θ --------
    #     print(f"\n[阶段 2] 训练扩散模型 θ (直到收敛, φ 固定)")

    #     patience = getattr(self.config, 'diff_patience', 5)   # 连续多少次没提升就停
    #     min_delta = getattr(self.config, 'diff_min_delta', 1e-2)  # 最小改善幅度
    #     best_loss = float('inf')
    #     bad_count = 0
    #     epoch = 0

    #     while True:
    #         loss = self._train_epoch_theta(train_data, epoch)
    #         if verbose:
    #             self.logger.info(f"[阶段 2] Epoch {epoch+1} | Diff Loss={loss:.4f}")

    #         # 验证
    #         if valid_data and (epoch + 1) % self.eval_step == 0:
    #             self._do_validation(epoch, valid_data, test_data, saved, verbose)

    #         # 收敛判断
    #         if best_loss - loss > min_delta:
    #             best_loss = loss
    #             bad_count = 0
    #         else:
    #             bad_count += 1

    #         if bad_count >= patience:
    #             print(f"θ 训练在 {epoch+1} 轮时提前收敛 ✅ (最优 Loss={best_loss:.4f})")
    #             break

    #         epoch += 1


    #     # -------- 阶段 3: 联合阶段 --------
    #     print(f"\n[阶段 3] 联合训练 φ+θ ({E3} 轮, θ 固定)")
    #     for epoch in range(E3):
    #         loss = self._train_epoch_joint(train_data, epoch)
    #         if verbose:
    #             self.logger.info(f"[阶段 3] Epoch {epoch+1}/{E3} | Joint Loss={loss:.4f}")
    #         if valid_data and (epoch + 1) % self.eval_step == 0:
    #             self._do_validation(epoch, valid_data, test_data, saved, verbose)

    #     print("\n===> 三阶段训练完成 ✅")
    #     return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    def _do_validation(self, epoch, valid_data, test_data, saved=True, verbose=True):
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result.get(self.valid_metric, valid_result.get('NDCG@20', 0.0))
        test_result = self.evaluate(test_data, is_test=True) if test_data else None

        # 判断是否更新 best
        is_better = (valid_score > self.best_valid_score) if self.valid_metric_bigger else (valid_score < self.best_valid_score)
        if is_better:
            self.best_valid_score = valid_score
            self.best_valid_result = valid_result
            self.best_test_upon_valid = test_result
            if saved:
                save_path = f'best_model_{getattr(self.config, "model", "GDNSM")}.pth'
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f'Best model saved to {save_path}')

        if verbose:
            self.logger.info(f"Epoch {epoch} | Valid: {valid_score:.4f}")
            if test_result:
                self.logger.info(f"Test Result: {dict2str(test_result)}")
    def fit(self, train_data, valid_data=None, test_data=None, saved=True, verbose=True):
        """
        每个 epoch 都执行三个阶段:
        1. 更新推荐模型 φ (BPR+CL)
        2. 更新扩散模型 θ (φ 冻结)
        3. 联合训练 (θ 固定, φ 更新, 生成负样本)
        """
        E = getattr(self.config, 'total_epochs', 100)  # 总 epoch 数
        self.best_valid_score = -1e9 if self.valid_metric_bigger else 1e9
        self.best_valid_result, self.best_test_upon_valid = {}, {}

        print(f"\n===> 开始联合训练 (每个 epoch 包含 3 个阶段)，总轮数 {E}")

        for epoch in range(E):
            # -------- 阶段 1: 推荐模型 φ --------
            loss_phi = self._train_epoch_phi(train_data, epoch)
            
            # -------- 阶段 2: 扩散模型 θ --------
            loss_theta = self._train_epoch_theta(train_data, epoch)

            # -------- 阶段 3: 联合训练 --------
            loss_joint = self._train_epoch_joint(train_data, epoch)

            if verbose:
                self.logger.info(
                    f"Epoch {epoch+1}/{E} | "
                    f"Phi Loss={loss_phi:.4f} | "
                    f"Theta Loss={loss_theta:.4f} | "
                    f"Joint Loss={loss_joint:.4f}"
                )

            # 验证
            if valid_data and (epoch + 1) % self.eval_step == 0:
                self._do_validation(epoch, valid_data, test_data, saved, verbose)

        print("\n===> 训练完成 ✅")
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid





    def plot_train_loss(self, save_path=None):
        epochs = sorted(self.train_loss_dict.keys())
        losses = [self.train_loss_dict[e] for e in epochs]
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.xticks(epochs)
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
