# trainer.py
import torch
from tqdm import tqdm
# 假设你会在 utils 里写 sampler 和 scheduler
from utils.sampler import DiffusionSampler 
from utils.scheduler import dynamic_difficulty_scheduler
from utils.topk_evaluator import TopKEvaluator # 导入新的评估器
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model, train_dataloader, valid_dataloader, optimizer):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.device = config['device']
        self.evaluator = TopKEvaluator(config)
        
        # 初始化 Diffusion Sampler (对应论文 2.4.2 节)
        self.sampler = DiffusionSampler(model.diffusion_model, config)

    def _train_epoch(self, epoch_idx):
        self.model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 创建一个进度条
        for batch_idx, batch_data in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}")):
            # --- 1. 数据解包与移动 (根据侦察笔记修改) ---
            # batch_data 的形状是 (3, batch_size)
            batch_data = batch_data.to(self.device) # 先把整个张量移动到设备
        
            users = batch_data[0, :]
            pos_items = batch_data[1, :]
            neg_items = batch_data[2, :]

            # --- 2. PyTorch 训练黄金五步法 (保持不变) ---
            # === Algorithm 1, lines 7-12: 训练扩散模型 ===
            self.optimizer.zero_grad()
            # 注意：论文中是两个独立的优化步骤，这里为了简化先合并，后续可拆分
            # diff_loss = self.model.calculate_diffusion_loss(pos_items) # 在模型主类中实现
            # diff_loss.backward()
            # self.optimizer.step()

            # === Algorithm 1, lines 13-26: 训练推荐模型 (MCI Encoder) ===
            self.optimizer.zero_grad()

            #针对这里的loss计算到底用哪一种，进行讨论
            # # 1. 生成困难负样本 (lines 13-21)
            # num_generated = dynamic_difficulty_scheduler(epoch_idx, self.config)
            # generated_negs_embeds = None
            # if num_generated > 0:
            #     # 这一步需要从正样本获取条件信息
            #     with torch.no_grad():
            #         user_embeds, _, _ = self.model.mci_encoder(users, [])
            #         # ... 获取 s_t, s_v ...
            #         # generated_negs_embeds = self.sampler.generate(...)
            #         pass # 暂时留空

            # # 2. 计算各项损失 (lines 4-5, 22-24)
            # # 你需要在 GDNSM 主模型类中实现一个 calculate_loss 方法
            # bpr_loss, cl_loss, neg_loss, diff_loss = self.model.calculate_loss(
            #     users, pos_items, neg_items, generated_negs_embeds, num_generated
            # )
            
            # # 3. 汇总损失并反向传播 (line 25)
            # # loss = bpr_loss + self.config['lambda_cl'] * cl_loss + self.config['lambda_neg'] * neg_loss + diff_loss
            # loss = bpr_loss + self.config['lambda_cl'] * cl_loss # 先从简单的开始
            
            loss = self.model.calculate_loss(batch_data)

            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()

            # --- 冒烟测试 (保持) ---
            if batch_idx >= 0:
                print("\nSmoke test passed for one batch!")
                break

        return epoch_loss / len(self.train_dataloader)

    def fit(self):
        """
        Drives the complete training and validation loop, with integrated negative sampling
        at the beginning of each epoch.
        """
        logger.info("================== Starting Training ==================")
        for epoch_idx in range(self.config['epochs']):
            
            # --- [核心修改] 在每个 epoch 开始前，执行负采样 ---
            logger.info(f"Epoch {epoch_idx} | Phase: Negative Sampling")
            # 我们通过 train_dataloader 访问其内部的 dataset 对象 (即 TrnData 实例)
            # 然后调用它的 negSampling 方法
            self.train_dataloader.dataset.negSampling()
            logger.info("Negative sampling for epoch completed.")

            # --- 训练阶段 ---
            train_loss = self._train_epoch(epoch_idx)
            # self._log_metrics({'train_loss': train_loss}, epoch_idx, 'train') # 假设有 log 方法

            # --- 评估阶段 ---
            # 检查是否到达评估的 epoch
            if (epoch_idx + 1) % self.config['eval_step'] == 0:
                logger.info(f"Epoch {epoch_idx} | Phase: Validation")
                # 调用我们已经写好的评估方法
                valid_score = self._valid_epoch(epoch_idx)
                
                # 可以在这里添加模型保存和早停的逻辑
                # ...
        
        logger.info("================== Training Finished ==================")

    # def evaluate(self):
    #     # ... 评估逻辑 ...
    #     pass

# common/trainer.py

    @torch.no_grad()
    def _valid_epoch(self, epoch_idx):
        """
        内部方法，负责协调和驱动一轮完整的、两阶段的验证。
        """
        self.model.eval() # 切换到评估模式
        
        # --- 阶段一：收集所有批次的 Top-K 结果 ---
        
        batch_matrix_list = [] # 用于存储每个批次的 topk_index
        
        # 遍历验证数据加载器
        for batch_idx, batch_data in enumerate(tqdm(self.valid_dataloader, desc=f"Epoch {epoch_idx} | Validation")):
            # 假设 valid_dataloader 每次返回一个批次的用户ID
            users_batch = batch_data.to(self.device)
            
            # 1. 调用模型生成当前批次用户的分数
            scores_batch = self.model.full_sort_predict(users_batch)
            
            # 2. 调用评估器的 collect 方法处理当前批次的结果
            # 注意：第二个参数 interaction 可以暂时用一个简单的对象或 None 替代
            # 因为在 full sort 模式下，它可能不是必需的。需要根据具体实现调整。
            # 这里我们假设它不需要 interaction 参数。
            topk_index_batch = self.evaluator.collect(None, scores_batch, full=True)
            
            # 3. 将当前批次的结果存入列表
            batch_matrix_list.append(topk_index_batch)
            
        # --- 阶段二：调用评估器的 evaluate 方法进行最终计算 ---
        
        # 将 valid_dataloader 自身作为 eval_data 传入
        results = self.evaluator.evaluate(batch_matrix_list, self.valid_dataloader)

        # --- 记录和返回关键指标 (逻辑保持不变) ---
        key_metric = self.config['valid_metric']
        metric_name, k = key_metric.split('@')
        k = int(k)
        final_score = results[f'{metric_name.lower()}@{k}'] # 注意 evaluator 返回的是小写 key
        
        self._log_metrics(results, epoch_idx, 'valid')
        
        return final_score

