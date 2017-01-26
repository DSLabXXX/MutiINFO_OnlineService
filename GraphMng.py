# -*- coding:utf-8 -*-
__author__ = 'c11tch'
from InfRank import InfRank
from scipy.sparse import lil_matrix, coo_matrix, eye, vstack, hstack
from queue import Queue
import numpy as np
import time
import logging
import logging.config


class GraphMng:

    def __init__(self):
        # 0*0的稀疏矩陣
        self.matrix_II = lil_matrix((0, 0))
        self.matrix_IT = lil_matrix((0, 0))
        self.matrix_TI = lil_matrix((0, 0))
        self.matrix_TT = lil_matrix((0, 0))

        # 建造上述矩陣用的個別的list
        self.list_IT_iid = []
        self.list_IT_tid = []
        self.list_IT_rat = []

        self.matrixTrans = lil_matrix((0, 0))

        self.valItemsNum = 0
        self.valTagsNum = 0
        self.valItemNum_MatrixTrans = 0
        self.valTagNum_MatrixTrans = 0

        self.valII = 1./2
        self.valTI = 1./2
        self.valIT = 1./2
        self.valTT = 1./2

        self.valQueryItem = 0.1
        self.valQueryTag = 0.9

        self.valSigmoidScale = 30
        self.valIterationNum = 30
        self.valDiffusionRate = 0.8
        self.valInfDiffRate = 0.8
        self.valDampFactor = 0.5
        self.valInfDampFactor = 0.5

        self.objInfRank = InfRank(self.valInfDampFactor, self.valDampFactor, self.valInfDiffRate, self.valIterationNum)
        """ For check no one using this graph """
        self.userQueue = Queue()

        self.log = logging.getLogger('[OSS].[GM]')

    """ ---------------------------------- Parameters modification ---------------------------------- """
    def upd_trans_param(self, vii, vti, vit, vtt):
        self.valII = vii
        self.valTI = vti
        self.valIT = vit
        self.valTT = vtt

    def show_mat_state(self):
        self.log.info('current num of items:{0} tags:{1}'.format(self.valItemsNum, self.valTagsNum))
        # self.log.info('current state of list_IT_iid:{0}'.format(self.list_IT_iid))
        # self.log.info('current state of list_IT_tid:{0}'.format(self.list_IT_tid))
        # self.log.info('current state of list_IT_rat:{0}'.format(self.list_IT_rat))

    """---------------------------------- Vertex management ---------------------------------- """

    """ Add a item to the graph """
    def add_new_item(self, userID):
        self.valItemsNum += 1

    def make_item_mat(self):
        self.matrix_II = lil_matrix((self.valItemsNum, self.valItemsNum)).tocsr()

    """ Add an tag to the graph """
    def add_new_tag(self, tagID):
        self.valTagsNum += 1

    def make_tag_mat(self):
        self.matrix_TT = lil_matrix((self.valTagsNum, self.valTagsNum)).tocsr()

    """ ---------------------------------- Edge management ---------------------------------- """

    """ Add an edge at IT/TI matrices """
    def add_edge_i2t(self, iid, tid, rating):
        self.list_IT_iid.append(iid)
        self.list_IT_tid.append(tid)
        self.list_IT_rat.append(rating)

    """ ---------------------------------- Make All Matrices by Lists ---------------------------------- """

    """依據輸入結果完成所有矩陣。
            例如：根據AddEdgeUID2IID的陣列結果完成matrix_UI/IU
            shape是matUU大小*matII大小，UI / IU 則大小相反 """
    def make_all_mat(self):
        """ 做出基本II / TT 矩陣 """
        self.make_item_mat()
        self.make_tag_mat()

        """ 合併矩陣並轉為csr格式儲存 """
        """ UI / IU mat """
        self.matrix_IT = coo_matrix((self.list_IT_rat, (self.list_IT_iid, self.list_IT_tid)),
                                    shape=(self.matrix_II.shape[0], self.matrix_TT.shape[0])).tocsr()
        self.matrix_TI = coo_matrix((self.list_IT_rat, (self.list_IT_tid, self.list_IT_iid)),
                                    shape=(self.matrix_TT.shape[0], self.matrix_II.shape[0])).tocsr()

        """ 測試用，可刪除 """
        print('-------------------------- Make -- All -- Mat -----------------------')
        # print('UI', self.matrix_UI.todense())
        # print('IU', self.matrix_IU.todense())
        # print('UT',self.matrix_UT.todense())
        # print('TU',self.matrix_TU.todense())
        # print('IT',self.matrix_IT.todense())
        # print('TI',self.matrix_TI.todense())
        self.show_mat_state()

    """ ---------------------------------- Vertex management ---------------------------------- """

    """ --- Column-normalization --- """
    def col_normalize(self, matrix):
        val_vtx_id_max = matrix.shape[0]
        vec_one = np.ones((1, val_vtx_id_max))
        """ 取出每行的維度與index"""
        vec_col_degree = vec_one * matrix
        vec_val = vec_col_degree[vec_col_degree.nonzero()].tolist()
        (vecpos_x, vecpos_y) = vec_col_degree.nonzero()
        """ 將所有的值改為 1 / 維度  (PageRank )"""
        vec_val = [1./float(x) for x in vec_val]
        # print('[col_normalize]vec_val',vec_val)
        vecpos_y = vecpos_y.tolist()
        """ 做出對角矩陣 值為 1 / 每行的維度 """
        matrix_d = coo_matrix((vec_val, (vecpos_y, vecpos_y)), shape=(val_vtx_id_max, val_vtx_id_max))
        """ 將原來矩陣所有的值做 normalization ( 值 *  1 / 每行的維度 = 對於行的比重?) """
        # print('[col_normalize] matrixD',matrix_d.todense())
        matrix_ret = matrix * matrix_d
        # print('[col_normalize] matrixRet',matrix_ret.todense())
        return matrix_ret

    """ --- Row-normalization --- """
    def row_normalize(self, matrix):
        val_vtx_id_max = matrix.shape[1]
        vec_one = np.ones((val_vtx_id_max, 1))
        vec_row_degree = matrix * vec_one
        vec_val = vec_row_degree[vec_row_degree.nonzero()].tolist()
        (vecpos_x, vecpos_y) = vec_row_degree.nonzero()
        vec_val = [1./x for x in vec_val]
        vecpos_x = vecpos_x.tolist()

        matrix_d = coo_matrix((vec_val, (vecpos_x, vecpos_x)), shape=(val_vtx_id_max, val_vtx_id_max))

        matrix_ret = matrix_d * matrix
        return matrix_ret

    """ --- Symmetric-normalization --- (未建立)"""
    def symmetric_normalize(self, matrix):
        val_vtx_id_max = matrix.shape[1]
        vec_one = np.ones((val_vtx_id_max, 1))
        vec_row_degree = matrix * vec_one
        vec_val = vec_row_degree[vec_row_degree.nonzero()].tolist()
        (vecpos_x, vecpos_y) = vec_row_degree.nonzero()
        vec_val = [1. / (x ** 0.5) for x in vec_val]
        vecpos_x = vecpos_x.tolist()

        matrix_d = coo_matrix((vec_val, (vecpos_x, vecpos_x)), shape=(val_vtx_id_max, val_vtx_id_max))

        matrix_ret = matrix_d * matrix * matrix_d
        return matrix_ret

    """ sigmoid function """
    def sigmoid(self, s):
        theta = 1.0 / (1.0 + np.exp(-s))
        return theta

    """ sigmf Matrix將矩陣數值轉為0.5~1之間的值，若無相關則為0 """
    def sigmoid_mat(self, mat):
        vec_val = mat[mat.nonzero()]
        """ 非零的值超過1個才做，否則就使用原矩陣即可，使用.shape[1]避免矩陣格式不同而讀不到nnz的問題 """
        if vec_val.shape[1] > 0:
            vec_val = vec_val.tolist()[0]    # 轉為list後面才可計算sigmoid
            (vecpos_x, vecpos_y) = mat.nonzero()
            (m, n) = mat.shape
            vec_average = np.average(vec_val)
            val_sigmoid_scale = np.std(np.array(vec_val))
            """ avoid x/0 """
            if val_sigmoid_scale == 0:
                val_sigmoid_scale = 1
            else:
                """ remove outlier and calculate new std """
                std_vec_val = [x for x in vec_val if x < (val_sigmoid_scale + vec_average)]
                val_sigmoid_scale = np.std(np.array(std_vec_val))
                if val_sigmoid_scale == 0:
                    val_sigmoid_scale = 1
            """ sigmoid """
            vec_val = self.sigmoid(np.array(vec_val) / val_sigmoid_scale)
            mat = coo_matrix((vec_val, (vecpos_x, vecpos_y)), shape=(m, n))
        return mat

    """
          --- Update the transition matrix by normalizing the adjacent matrix.
          --- valOPcode has three value : {1 : ColumnNormalization; 2 : SymmetricNormalization; 3 : RowNormalization}
          --- Nete that processing this function may take much time, so update periodically """
    def upd_trans_mat(self):
        self.matrix_II = self.sigmoid_mat(self.matrix_II)
        self.matrix_TT = self.sigmoid_mat(self.matrix_TT)
        self.matrix_IT = self.sigmoid_mat(self.matrix_IT)
        self.matrix_TI = self.sigmoid_mat(self.matrix_TI)

        """ 將矩陣乘以各自的權重後合併
           [ matrix_II ] [ matrix_IT ]
           [ matrix_TI ] [ matrix_TT ]
        """
        temp_row_i = hstack([self.matrix_II * self.valII, self.matrix_IT * self.valIT])
        temp_row_t = hstack([self.matrix_TI * self.valTI, self.matrix_TT * self.valTT])
        self.matrixTrans = vstack([temp_row_i, temp_row_t])
        self.log.debug('[upd_trans_mat] matrixTrans: {0}'.format(self.matrixTrans.shape))
        # print(self.matrixTrans.todense())

        """ 使用真矩陣 """
        self.valItemNum_MatrixTrans = self.valItemsNum
        self.valTagNum_MatrixTrans = self.valTagsNum
        """ 使用真矩陣 """

        val_vertex_id_max = self.matrixTrans.shape
        matrix_i = eye(val_vertex_id_max[0], val_vertex_id_max[1])

        """ Column-Normalization (for PageRank) """
        self.objInfRank.matrixColNormTrans = self.valDiffusionRate * self.col_normalize(self.matrixTrans) + (1 - self.valDiffusionRate) * matrix_i
        """ Symmetric Normalization (for InfRank) """
        self.objInfRank.matrixSymNormTrans = self.valDiffusionRate * self.symmetric_normalize(self.matrixTrans) + (1 - self.valDiffusionRate) * matrix_i

        # clear matrixI;
        # clear obj.matrixTrans;
        """ 以下暫時不用 """
        # matrixRowNorm = self.RowNomalization(self.matrixTrans)
        # # self.objInfRank = self.objInfRank.InfluenceProcess(matrixRowNorm)
        # self.objInfRank.InfluenceProcess(matrixRowNorm)

    """ 處理Query 資料以及推薦排序 """
    def exec_ranking_process(self, vec_query_iid, vec_query_tid, rlt_type, norm_type, k):
        val_query_item_size = len(vec_query_iid)
        val_query_tag_size = len(vec_query_tid)

        self.log.debug('[ExecRankingProcess] ItemNum {0}'.format(self.valItemNum_MatrixTrans))
        self.log.debug('[ExecRankingProcess] TagNum {0}'.format(self.valTagNum_MatrixTrans))
        val_trans_mat_size = self.valItemNum_MatrixTrans + self.valTagNum_MatrixTrans

        """  Start to Construct the Query vector of initial state : vecInitialState  """
        vec_init_idx = np.zeros((val_query_item_size + val_query_tag_size))
        vec_init_value = np.zeros((val_query_item_size + val_query_tag_size))

        for i in range(val_query_item_size):
            vec_init_idx[i] = vec_query_iid[i]
            vec_init_value[i] = self.valQueryItem * 1 / val_query_item_size

        for i in range(val_query_tag_size):
            vec_init_idx[val_query_item_size + i] = self.valItemNum_MatrixTrans + vec_query_tid[i]
            vec_init_value[val_query_item_size + i] = self.valQueryTag * 1 / val_query_tag_size

        vec_init_state = coo_matrix((vec_init_value.tolist(),
                                     (vec_init_idx.tolist(), np.zeros(len(vec_init_value)).tolist())),
                                    shape=(val_trans_mat_size, 1)).tocsr()
        self.vec_init_state = vec_init_state / vec_init_state.sum()  # normalization
        """  Finished Construct the vector of initial state : vecInitialState  """

        val_pr_start = time.time()
        self.vecInfRankVertexScore = self.objInfRank.DiffusionProcess(self.vec_init_state, self.vec_init_state, norm_type)
        val_pr_stop = time.time()
        self.log.warning('[ExecRankingProcess] PR process time= {0}'.format(val_pr_stop-val_pr_start))

        """ Starting Output the result from 'vecInfRankVertexScore': rank_rlt_list  """
        if rlt_type == 1:
            """ user """
            """ 只取使用者這一段來處理 """
            vec_rlt_score = self.vecInfRankVertexScore.toarray()[0:self.valItemNum_MatrixTrans]
            """ 取得排序後的index """
            self.vecSortedIndex_InfRank = sorted(range(len(vec_rlt_score)), reverse=True, key=lambda k: vec_rlt_score[k])
            if val_query_item_size > 0:
                rank_rlt_list = self.vecSortedIndex_InfRank[0+val_query_item_size: min(k+val_query_item_size, self.valItemNum_MatrixTrans)]
            else:
                rank_rlt_list = self.vecSortedIndex_InfRank[0: min(k, self.valItemNum_MatrixTrans)]
        elif rlt_type == 2:
            """ item """
            vec_rlt_score = self.vecInfRankVertexScore.toarray()[self.valItemNum_MatrixTrans: self.valItemNum_MatrixTrans+self.valTagNum_MatrixTrans]
            """ 取得排序後的index """
            self.vecSortedIndex_InfRank = sorted(range(len(vec_rlt_score)), reverse=True, key=lambda k: vec_rlt_score[k])

            if val_query_tag_size > 0:
                rank_rlt_list = self.vecSortedIndex_InfRank[0+val_query_tag_size: min(k+val_query_tag_size, self.valTagNum_MatrixTrans)]
            else:
                rank_rlt_list = self.vecSortedIndex_InfRank[0: min(k, self.valTagNum_MatrixTrans)]
        """ Finish Output the result from 'vecInfRankVertexScore': rank_rlt_list  """
        # print('看分數:\n', vec_rlt_score)
        # print('看排序後分數:\n', vec_rlt_score[rank_rlt_list])  # 看分數
        return rank_rlt_list

    def upd_param(self, valIterationNum, valDampFactor, valDiffusionRate, valInfDiffRate, valInfDampFactor):
        self.valIterationNum = valIterationNum
        self.valDampFactor = valDampFactor
        self.valDiffusionRate = valDiffusionRate
        self.valInfDiffRate = valInfDiffRate
        self.valInfDampFactor = valInfDampFactor
        self.objInfRank = self.objInfRank.ModifyParam(valInfDampFactor, valDampFactor,
                                                      valDiffusionRate, valIterationNum)
