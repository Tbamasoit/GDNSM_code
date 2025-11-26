# trainer.py
import torch
from tqdm import tqdm
import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter # <--- 核心组件
from utils.scheduler import DynamicDifficultyScheduler 
from utils.topk_evaluator import TopKEvaluator # 导入新的评估器
from types import SimpleNamespace # <--- 新增这个引用

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
        self.epoch_diff_loss = 0.0
        self.epoch_rec_loss = 0.0
        # 初始化调度器
        self.scheduler = DynamicDifficultyScheduler(config)
        
        # 初始化 Diffusion Sampler (对应论文 2.4.2 节)
        # self.sampler = DiffusionSampler(model.diffusion_model, config)
        # === [新增] 初始化 TensorBoard Writer ===
        # 为了防止不同实验的日志覆盖，我们用 "当前时间" 或 "配置参数" 做子文件夹名
        timestamp = time.strftime('%m%d_%H%M')
        exp_name = f"{config['dataset']}_d{config['d_epoch']}_lam{config['lambda_neg']}_{timestamp}"
        log_dir = os.path.join(config['tensorboard_log_dir'], exp_name)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0 # 用于记录总的 Batch 数
        
        logger.info(f"TensorBoard initialized. Logs will be saved to: {log_dir}")


    def _train_epoch(self, epoch_idx):
        self.model.train()
        epoch_total_loss = 0.0
        
        # 使用 tqdm 创建一个进度条
        for batch_idx, batch_data in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}")):
            # --- 1. 数据解包与移动 (根据侦察笔记修改) ---
            # batch_data 的形状是 (3, batch_size)
            
            # --- [核心修改] 数据解包与移动 ---
            # batch_data 是一个列表: [users, pos_items, neg_items]
            # 我们不能直接 batch_data.to(device)，必须逐个移动
            users, pos_items, neg_items = batch_data
            
            users = users.to(self.device).long()
            pos_items = pos_items.to(self.device).long()
            neg_items = neg_items.to(self.device).long()

            # 重新打包成列表，以便传给 calculate_loss
            interaction = [users, pos_items, neg_items]

            # batch_data = batch_data.to(self.device) # 先把整个张量移动到设备
        
            # users = batch_data[0, :]
            # pos_items = batch_data[1, :]
            # neg_items = batch_data[2, :]

            # ======================================================================
            # 阶段一: 训练扩散模型 (对应 Algorithm 1, lines 7-12)
            # ======================================================================

            # 1. 准备数据：获取正样本嵌入和条件信息
            # 我们需要模型提供一个方法来获取这些东西
            with torch.no_grad():
                pos_item_embeds, diffusion_conditions = self.model.get_diffusion_inputs(users, pos_items)

            # 2. 循环训练扩散模型 N 次 (这里设为 3)
            # 这能让扩散模型更快地适应 Item Embedding 的变化，生成更精准的样本
            for _ in range(3):
                self.optimizer.zero_grad()
                diffusion_loss = self.model.diffusion_MM(pos_item_embeds, diffusion_conditions, self.device)
                # diffusion_loss = self.model.diffusion_MM(pos_item_embeds, diffusion_conditions, self.device)
                
                # 3. 反向传播并更新 (只更新扩散模型的参数)
                # 为了实现交替训练，理想情况下应该有两个优化器。
                # 简化版：我们先用一个优化器更新所有参数。
                diffusion_loss.backward()
                
                # (可选) 如果你之前加了梯度裁剪，这里也可以加，通常扩散模型比较稳，不加也行
                # torch.nn.utils.clip_grad_norm_(self.model.diffusion_MM.parameters(), max_norm=1.0)
                
                self.optimizer.step()

            # ======================================================================
            # 阶段二: 训练推荐模型 (对应 Algorithm 1, lines 13-26)
            # ======================================================================

            # 1. [核心逻辑] 动态负样本生成与调度
            scheduled_negs = None

            # 只有当 Epoch 达到阈值，且 g(epoch) > 0 时才生成
            # 我们可以先问问 scheduler 需要多少个，如果需要0个，就别费劲生成了，省时间
            required_num = self.scheduler.get_g_epoch(epoch_idx)

            if required_num > 0:
                with torch.no_grad():
                    # 调用模型生成接口
                    neg_v, neg_t, neg_vt = self.model.generate_batch_negatives(users, pos_items)
                    
                    # 调用调度器筛选
                    scheduled_negs = self.scheduler.schedule(epoch_idx, neg_v, neg_t, neg_vt)

            # 2. 计算推荐 Loss
            self.optimizer.zero_grad()
            # 将筛选后的负样本传入 calculate_loss
            recommender_loss = self.model.calculate_loss(interaction,  generated_negs=scheduled_negs)
            # 3. 反向传播 (更新推荐模型)
            recommender_loss.backward()
            # [新增] 梯度裁剪，防止 Loss 爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # --- 记录总损失 ---
            # total_loss = diffusion_loss + recommender_loss
            # epoch_total_loss += total_loss.item()
            epoch_total_loss += (diffusion_loss.item() + recommender_loss.item())
            # 简单起见，你可以直接 print 出来调试
            if batch_idx % 20 == 0:
                logger.info(f"Batch {batch_idx}: Diff Loss={diffusion_loss.item():.4f}, Rec Loss={recommender_loss.item():.4f}")                                                 
            
            # === [新增] 记录 Step 级别的 Loss ===
            # 记录扩散模型 Loss
            self.writer.add_scalar('Loss/Diffusion_Step', diffusion_loss.item(), self.global_step)
            
            # 记录推荐模型 Loss
            self.writer.add_scalar('Loss/Recommendation_Step', recommender_loss.item(), self.global_step)
            
            # 记录总 Loss
            self.writer.add_scalar('Loss/Total_Step', diffusion_loss.item() + recommender_loss.item(), self.global_step)

            # === [进阶] 监控生成的负样本质量 (强烈推荐) ===
            # 这能帮你一眼看出生成的样本是不是太简单(得分低)或者太难(得分高)
            if scheduled_negs is not None:
                with torch.no_grad():
                    # 计算 User 和 Generated Neg 的相似度
                    # users 形状 [B], scheduled_negs 形状 [B, 1, D] -> [B, D]
                    # u_emb 需要重新获取一下或者在 calculate_loss 里返回，这里为了演示简化一下逻辑
                    # 如果你想做这个，需要在 calculate_loss 里把计算好的分数传出来，或者在这里简单算一下
                    pass 

            self.global_step += 1 # 步数 +1


            # --- 冒烟测试 (保持) ---
            if batch_idx >= 0 and self.config['smoke_test']:
                print("\nSmoke test passed for one batch!")
                break

        return epoch_total_loss / (batch_idx + 1)



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
            # users_batch = batch_data.to(self.device)
            users_batch = batch_data[0] # 取列表的第一个元素
            users_batch = users_batch.to(self.device).long()
            
            # 1. 调用模型生成当前批次用户的分数
            scores_batch = self.model.full_sort_predict([users_batch])
            
            # 2. 调用评估器的 collect 方法处理当前批次的结果
            # 注意：第二个参数 interaction 可以暂时用一个简单的对象或 None 替代
            # 因为在 full sort 模式下，它可能不是必需的。需要根据具体实现调整。
            # 这里我们假设它不需要 interaction 参数。
            # === [核心修复] 构造 Mock Interaction 对象 ===
            batch_size = users_batch.size(0)
            # 构造一个带有 user_len_list 属性的对象
            # 在 Full Sort 模式下，通常认为每个 batch 里的每个位置对应 1 个用户
            interaction = SimpleNamespace(
                user_len_list=[1] * batch_size,
                pos_len_list=[1] * batch_size  #有些 evaluator 可能也需要这个，顺手加上保险
            )
            # 2. 收集 TopK
            # 将构造好的 interaction 传进去，而不是 None
            topk_index_batch = self.evaluator.collect(interaction, scores_batch, full=True)
            
            # 3. 将当前批次的结果存入列表
            batch_matrix_list.append(topk_index_batch)

            # 冒烟测试
            if self.config['smoke_test']:
                break
            
        # --- 阶段二：调用评估器的 evaluate 方法进行最终计算 ---
        
        # 将 valid_dataloader 自身作为 eval_data 传入
        # === [核心修复] 传入 dataset 而不是 dataloader ===
        # 注意：这里需要传入原始的 valid_dataloader 对象，因为 evaluator 需要用到它的 ground truth
        eval_data = self.valid_dataloader.dataset  # 获取底层的数据集对象
        results = self.evaluator.evaluate(batch_matrix_list, eval_data)

        # --- 记录和返回关键指标 (逻辑保持不变) ---
        key_metric = self.config['valid_metric']
        metric_name, k = key_metric.split('@')
        k = int(k)
        # 处理大小写兼容性 (evaluator 通常返回小写 key，如 recall@20)
        result_key = f'{metric_name.lower()}@{k}'
        final_score = results.get(result_key, 0.0) # 注意 evaluator 返回的是小写 key
        # 简单打印一下验证结果
        logger.info(f"Validation Results (Epoch {epoch_idx}): {results}")

        # self._log_metrics(results, epoch_idx, 'valid')
        
        return final_score

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
            # logger.info(f"Epoch {epoch_idx} | Train Loss: {train_loss:.4f}")
            # self._log_metrics({'train_loss': train_loss}, epoch_idx, 'train') # 假设有 log 方法
            # === [新增] 记录 Epoch 级别的 Train Loss ===
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch_idx)


            # --- 评估阶段 ---
            # 检查是否到达评估的 epoch
            if (epoch_idx + 1) % self.config['eval_step'] == 0:
                # 调用我们已经写好的评估方法
                valid_score = self._valid_epoch(epoch_idx)
                logger.info(f"Epoch {epoch_idx} | Valid Score ({self.config['valid_metric']}): {valid_score:.4f}")

                # === [新增] 记录验证集指标 ===
                metric_name = self.config['valid_metric'] # e.g. Recall@20
                self.writer.add_scalar(f'Metric/{metric_name}', valid_score, epoch_idx)
                # 如果你想记录更多，比如 NDCG@10，可以在 _valid_epoch 里把 results 返回出来
                # self.writer.add_scalar('Metric/NDCG@10', results['ndcg@10'], epoch_idx)

                # 可以在这里添加模型保存和早停的逻辑
                # ...

            # 冒烟测试：只跑一个 Epoch 就退出整个训练
            if self.config['smoke_test']:
                logger.info("💨 Smoke test finished. Stopping training.")
                break
        
        # === [新增] 训练结束关闭 Writer ===
        self.writer.close()
        logger.info("================== Training Finished ==================")

    # def evaluate(self):
    #     # ... 评估逻辑 ...
    #     pass
