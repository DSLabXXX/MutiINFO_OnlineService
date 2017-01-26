# -*- coding:utf-8 -*-
__author__ = 'c11tch'
from scipy.sparse import lil_matrix,coo_matrix,eye,vstack,hstack
import numpy as np


class InfRank:
    def __init__(self, valInfDampFactor, valDampFactor, valDiffRate, itNum):
        self.valDMPDampFactor = valDampFactor
        self.iterationNum = itNum
        self.valDiffusionRate = valDiffRate
        self.valInfDampFactor = valInfDampFactor
        self.matrixSymNormTrans = lil_matrix((0, 0))

    def ModifyParam(self, valInfDampFactor, valDampFactor, valDiffRate, itNum):
        self.valDMPDampFactor = valDampFactor
        self.iterationNum = itNum
        self.valDiffusionRate = valDiffRate
        self.valInfDampFactor = valInfDampFactor

    """ ------------------------ Update the column-normalized transition matrix ----------------------- """

    def UpdateColNormTransMatrix(self, matrix):
        self.matrixColNormTrans = matrix

    def UpdateSymNormTransMatrix(self, matrix):
        self.matrixSymNormTrans = matrix

    """  影響力的處理 """
    def InfluenceProcess(self, matrixA):
        if self.matrixSymNormTrans.shape[0] != 0:
            valVeretxIDMax = self.matrixSymNormTrans.shape[0]
        else:
            valVeretxIDMax = self.matrixColNormTrans.shape[0]

        vecDiffusion = np.ones((1, valVeretxIDMax)) * self.valInfDampFactor
        vecDiffperIt = vecDiffusion

        step = 20
        for i in range(step):
            vecDiffperIt = (1 - self.valInfDampFactor) * vecDiffperIt * matrixA
            vecDiffusion = vecDiffusion + vecDiffperIt

        self.vecDiff = vecDiffusion.T

    """ Personalize Page Rank  """
    def DiffusionProcess(self, vecHDRankPreference, vecInitialState, typeID):
        iteration = 0
        vecInfRankVertexScore = vecInitialState
        matrixA = self.matrixColNormTrans

        """ Page Rank """
        valPageSteps = 20

        while iteration <= valPageSteps:
            vecInfRankVertexScore = (1-self.valDMPDampFactor) * (matrixA * vecInfRankVertexScore) + self.valDMPDampFactor * vecInitialState
            iteration += 1

        if float(typeID) == 1:    # --- Use column-normalized matrix ---
            matrixA = self.matrixColNormTrans
        elif float(typeID) == 2:    # --- Use symmetric normalized matrix ---
            matrixA = self.matrixSymNormTrans

        valVeretxIDMax = matrixA.shape[0]
        vecHDRankPreference = vecInfRankVertexScore

        """ 以下為加上影響力處理的 Inf Rank 因暫時不會用到故註解掉 """
        # vecDiffusion = self.vecDiff
        #
        # valDifference = -1
        # iteration = 0
        #
        # # print vecInfRankVertexScore# sparse matrix(array)
        # # print vecDiffusion# nd array
        # while valDifference == -1 or iteration <= self.iterationNum:
        #     # [max Index * 1]
        #     valPreState = vecInfRankVertexScore[0, 0]
        #
        #     # --- Heuristic function ---
        #     """ array 用 a* b  matrix 用 multiply(a, b) -> 對應項相乘
        #                     array 用 dot(a, b) matrix 用 a* b -> 矩陣or向量相乘(維度需相對應) """
        #     vecHeuristic = vecInfRankVertexScore.toarray() * vecDiffusion
        #     """ 如果 上面是array 型別 應該是對的 這部分寫好再測 """
        #     vecInfHeuristic = vecHeuristic
        #
        #     # --- Calculate Dt : for normalization ---
        #     vecVal = vecInfHeuristic[vecInfHeuristic.nonzero()]
        #     if vecVal.shape[1] > 0:
        #         vecVal = vecVal.tolist()[0]    # 轉為list後面才可計算sigmoid
        #         (vecPosX, vecPosY) = vecInfHeuristic.nonzero()
        #         (m, n) = vecInfHeuristic.shape
        #     matrixHeuristic = coo_matrix((vecVal, (vecPosX, vecPosY)), shape=(valVeretxIDMax, valVeretxIDMax))
        vecInfRankVertexScore = vecHDRankPreference
        return vecInfRankVertexScore