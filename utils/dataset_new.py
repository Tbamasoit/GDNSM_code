import pickle
import numpy as np
from scipy.sparse import coo_matrix
# from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
#add
import os
# from Params import args # 我们稍后会替换掉它
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
import lmdb

class DataHandler:
    def __init__(self):
        if args.data == 'baby':
            predir = './Datasets/baby/'
        elif args.data == 'sports':
            predir = './Datasets/sports/'
        elif args.data == 'tiktok':
            predir = './Datasets/tiktok/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'

        self.imagefile = predir + 'image_feat.npy'
        self.textfile = predir + 'text_feat.npy'
        if args.data == 'tiktok':
            self.audiofile = predir + 'audio_feat.npy'

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
            # ret = pickle.load(fs)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def loadFeatures(self, filename):
        feats = np.load(filename)
        return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

    def LoadData(self):
        trnMat = self.loadOneFile(self.trnfile)
        tstMat = self.loadOneFile(self.tstfile)
        self.trnMat = trnMat
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

        self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
        self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
        if args.data == 'tiktok':
            self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)

        self.diffusionData = DiffusionData(torch.FloatTensor(self.trnMat.A))
        self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)
#delete


class TrnData(data.Dataset):
    def __init__(self, config, coomat):
        self.config = config
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.config['n_items'])
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, config, coomat, trnMat):
        self.config = config
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
    
    # === [新增] 核心修复：提供评估所需的 Ground Truth ===
    def get_eval_items(self):
        """
        返回测试集中每个用户的真实正样本列表。
        注意：返回顺序必须与 DataLoader 遍历用户的顺序一致 (即 self.tstUsrs 的顺序)。
        """
        ground_truth = []
        for u in self.tstUsrs:
            # tstLocs[u] 存储了用户 u 在测试集中的所有交互物品 ID
            ground_truth.append(self.tstLocs[u])
        return ground_truth
    
    def get_eval_len_list(self):
        """
        [新增] 返回测试集中每个用户的真实正样本数量。
        用于计算 Recall 等指标的分母。
        """
        len_list = []
        for u in self.tstUsrs:
            # self.tstLocs[u] 是用户 u 的真实交互物品列表
            len_list.append(len(self.tstLocs[u]))
        return np.array(len_list)


class DiffusionData(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return item, index
    
    def __len__(self):
        return len(self.data)

class MyDataset:
    def __init__(self, config):
        #print("MyDataset is being initialized!")
        self.config = config
        self.device = config['device']
        # 1. 定义文件路径 (用 config 替换 args)
        #self.predir = self.config['data_path'] # 我们假设 data_path 就是 baby/ 或 sports/ 的完整路径
        self.predir = os.path.join(self.config['data_path'], self.config['dataset'])
        self.trnfile = os.path.join(self.predir, 'trnMat.pkl')
        self.tstfile = os.path.join(self.predir, 'tstMat.pkl')
        self.imagefile = os.path.join(self.predir, 'image_feat.npy')
        self.textfile = os.path.join(self.predir, 'text_feat.npy')
        # 2. 加载交互矩阵
        print("Loading interaction matrices...")
        self.trnMat = self._loadOneFile(self.trnfile)
        self.tstMat = self._loadOneFile(self.tstfile)
        print("Interaction matrices loaded.")
        # 3. 设置用户/物品数量 (存入 config，供全局使用)
        self.config['n_users'], self.config['n_items'] = self.trnMat.shape

        # 并且，MyDataset 也应该有 self.n_users 和 self.n_items 属性
        self.n_users = self.config['n_users']
        self.n_items = self.config['n_items']
        # 4. 构建邻接矩阵
        print("Building adjacency matrix...")
        #self.torchBiAdj = self._makeTorchAdj(self.trnMat)
        # 步骤 4.1: 调用我们新迁移的方法，生成 scipy 格式的邻接矩阵
        adj_mat_scipy = self._get_adj_mat()
        # 步骤 4.2: 调用我们新迁移的工具函数，将 scipy 矩阵转换为 PyTorch 稀疏张量
        self.torchBiAdj = self._sparse_mx_to_torch_sparse_tensor(adj_mat_scipy)
        # 步骤 4.3: _get_adj_mat 内部会设置 self.R，我们也需要把它转换成张量
        # 因为 GDNSM 的 forward 方法里用到了 self.R
        self.R_tensor = self._sparse_mx_to_torch_sparse_tensor(self.R)
        print("Adjacency matrix built.")
        # 5. 加载特征
        print("Loading features...")
        self.image_feats, self.config['image_feat_dim'] = self._loadFeatures(self.imagefile)
        self.text_feats, self.config['text_feat_dim'] = self._loadFeatures(self.textfile)
        print("Features loaded.")   
        
        # --- [新增] 步骤 5: 构建或加载多模态相似度图 ---
        print("Building/Loading modality graphs...")
        # 5.1 定义缓存文件的路径
        # 我们需要从 config 中获取 knn_k 和 sparse 参数
        self.knn_k = self.config['knn_k']
        self.sparse = True # 假设我们总是使用稀疏图
        
        dataset_root_path = os.path.join(self.config['data_path'], self.config['dataset'])
        image_adj_file = os.path.join(dataset_root_path, f'image_adj_{self.knn_k}_{self.sparse}.pt')
        text_adj_file = os.path.join(dataset_root_path, f'text_adj_{self.knn_k}_{self.sparse}.pt')

        # 5.2 处理视觉模态图
        if self.image_feats is not None: 
            # self.image_embedding = nn.Embedding.from_pretrained(self.image_feats, freeze=False)
            if os.path.exists(image_adj_file):
                print(f"Loading cached image graph from {image_adj_file}")
                image_adj = torch.load(image_adj_file)
            else:
                print("Building new image graph...")
                # image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_sim(self.image_feats) # 直接使用 self.image_feats
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym') # knn weighted each element before norm is float not {0, 1}
                print(f"Saving new image graph to {image_adj_file}")
                torch.save(image_adj, image_adj_file)
            # self.image_original_adj = image_adj.cuda()
            self.image_adj = image_adj.to(self.device)

        if self.text_feats is not None:
            # self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                print(f"Loading cached text graph from {text_adj_file}")
                text_adj = torch.load(text_adj_file)
            else:
                print("Building new text graph...")
                # [核心适配 1]
                text_adj = build_sim(self.text_feats) # 直接使用 self.text_feats
                # text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                print(f"Saving new text graph to {text_adj_file}")
                torch.save(text_adj, text_adj_file)
            # self.text_original_adj = text_adj.cuda()
            self.text_adj = text_adj.to(self.device)

        print("Modality graphs ready.")

        print("Creating PyTorch Dataset instances...")
        self.trn_dataset = TrnData(self.config, self.trnMat)
        self.tst_dataset = TstData(self.config, self.tstMat, self.trnMat) # TstData 可能也需要 config
        print("PyTorch Dataset instances created.")

        

        # if self.v_feat is not None:
        #     self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        # if self.t_feat is not None:
        #     self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)


    def _loadOneFile(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        # ret = pickle.load(fs)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def _normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def _makeTorchAdj(self, mat):
        # make ui adj
        a = sp.csr_matrix((self.config['n_users'], self.config['n_users']))
        b = sp.csr_matrix((self.config['n_items'], self.config['n_items']))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self._normalizeAdj(mat)

        # make cuda tensor
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def _loadFeatures(self, filename):
        feats = np.load(filename, allow_pickle=True)
        return torch.tensor(feats).float().cuda(), np.shape(feats)[1]
    
    def _LoadData(self):
        trnMat = self._loadOneFile(self.trnfile)
        tstMat = self._loadOneFile(self.tstfile)
        self.trnMat = trnMat
        args.user, args.item = trnMat.shape
        self.torchBiAdj = self._makeTorchAdj(trnMat)

        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

        self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
        self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
        if args.data == 'tiktok':
            self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)

        self.diffusionData = DiffusionData(torch.FloatTensor(self.trnMat.A))
        self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

    #来自原GDNSM的剪贴功能
    def _get_adj_mat(self): # [n_users, n_items]->[n_users+n_items, n_users+n_items]
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.trnMat.tolil()

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

    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_pytorch_datasets(self):
        """返回用于创建 DataLoader 的数据集实例"""
        return self.trn_dataset, self.tst_dataset

    def get_adj_matrix(self):
        """返回给 GNN 模型使用的邻接矩阵"""
        return self.torchBiAdj

    def get_all_features(self):
        """返回所有模态的特征张量"""
        # 暂时只返回图像和文本
        return self.image_feats, self.text_feats
    
    def get_dataset_info(self):
        return {
            'n_users': self.config['n_users'],
            'n_items': self.config['n_items'],
            'norm_adj': self.torchBiAdj,
            'R_tensor': self.R_tensor,
            'image_adj': self.image_adj,
            'text_adj': self.text_adj,
            'v_feat': self.image_feats,
            't_feat': self.text_feats
        }