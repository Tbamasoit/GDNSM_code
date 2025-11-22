# utils/scheduler.py
import math
import torch

class DynamicDifficultyScheduler:
    def __init__(self, config):
        """
        初始化动态难度调度器。
        
        参数 config 中必须包含:
            - d_epoch (int): Starting epoch S (论文中的 S)
            - lambda_coeff (float): 增长速率 (论文中的 lambda)
            - num_gen_samples (int): 每个难度级别生成的样本数 M
        """
        self.S = config['d_epoch']     # 起始 Epoch
        self.lam = config.get('lambda_coeff', 1.0) # 默认为 1.0
        self.M = config.get('num_gen_samples', 1)  # 默认为 1 (源码中似乎设为1或3)
        self.max_samples = 3 * self.M # 总共生成的样本池大小 (3个难度级)


    def get_g_epoch(self, epoch_idx):
        """
        计算当前 epoch 应该使用的负样本数量 g(epoch)
        公式: g(epoch) = int[ (epoch / S)^lambda * M ]
        """
        if epoch_idx < self.S:
            return 0
        
        ratio = (epoch_idx / self.S) ** self.lam
        # 论文逻辑：随着训练进行，逐渐纳入更多困难样本
        # 这里我们设定上限为生成的总数
        g_val = int(ratio * self.M) 
        
        # 限制范围 [0, 3*M]
        return min(g_val, self.max_samples)
    

    def schedule(self, epoch_idx, neg_v, neg_t, neg_vt):
        """
        输入生成的三个等级的负样本，返回筛选后的列表。
        顺序: Easy (V) -> Medium (T) -> Hard (VT)
        
        Args:
            epoch_idx: 当前 epoch
            neg_v, neg_t, neg_vt: 形状均为 [Batch, M, Dim] 的张量
        Returns:
            selected_negs: 形状为 [Batch, g(epoch), Dim] 的张量，如果 g=0 返回 None
        """
        g_count = self.get_g_epoch(epoch_idx)
        
        if g_count <= 0:
            return None
        
        # 1. 在 dim=1 (样本数量维度) 拼接
        # 结果形状: [Batch, 3*M, Dim]
        all_candidates = torch.cat([neg_v, neg_t, neg_vt], dim=1)
        
        # 2. 截取前 g_count 个
        selected_negs = all_candidates[:, :g_count, :]
        
        return selected_negs