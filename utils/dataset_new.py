import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
#add
import os
from Params import args # 我们稍后会替换掉它
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
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
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
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
        # 4. 构建邻接矩阵
        print("Building adjacency matrix...")
        self.torchBiAdj = self._makeTorchAdj(self.trnMat)
        print("Adjacency matrix built.")
        # 5. 加载特征
        print("Loading features...")
        self.image_feats, self.config['image_feat_dim'] = self._loadFeatures(self.imagefile)
        self.text_feats, self.config['text_feat_dim'] = self._loadFeatures(self.textfile)
        print("Features loaded.")       



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