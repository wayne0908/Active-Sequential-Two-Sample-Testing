import numpy as np 
import multiprocessing as mp
import pdb
import os
import pickle
import random
import typer 
import torch
import math
import time
from copy import deepcopy
from sklearn.model_selection import cross_val_score
import sys
from torch import nn
from skorch import NeuralNetClassifier
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import csr_matrix 
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC
from scipy.stats import t, chi2, norm, kendalltau, sem
from scipy.sparse.csgraph import minimum_spanning_tree
from modAL.models import ActiveLearner
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import f
from sklearn.calibration import CalibratedClassifierCV
from rpy2.robjects.packages import importr
from sklearn.cluster import KMeans
from rpy2 import robjects


class Torch_Model(nn.Module):
    def __init__(self, W):
        super(Torch_Model, self).__init__()
        self.W = W
        self.fcs = nn.Sequential(
                                nn.Linear(self.W,32),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(32,2),
        )

    def forward(self, x):
        out = x.float()
        out = out.view(-1,self.W)
        out = self.fcs(out)
        return out

def BuildNN(W=20):
	"""
	Neural network classifier
	W: int. Width of first layer
	"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	Cls = NeuralNetClassifier(Torch_Model(W),
							  criterion=nn.CrossEntropyLoss,
							  optimizer=torch.optim.Adam,
							  train_split=None,
							  verbose=0,
							  device="cpu")
	return Cls


def EnhanceUncertainty2(classifier, X_pool):
	Prob = classifier.predict_proba(X_pool)[:, -1]; 
	query_idx0 = np.argmax(Prob); query_idx1 = np.argmin(Prob)
	query_idx = [query_idx0, query_idx1]; 
	return query_idx, X_pool[query_idx]

def EnhanceUncertainty(classifier, X_pool):
	
	Prob = classifier.predict_proba(X_pool)[:, -1]; uncertainty = np.abs(Prob - 0.5)
	query_idx = np.argmax(uncertainty)
	return query_idx, X_pool[query_idx]


def SequentialPRT(args, StatsPath, QueryPath, Data, HoldoutData):
	from Utility import PlotStatsDistri2
	Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);
	if args.ReFillStats == 0:
		if args.debias ==0 or args.debias == 1 or args.debias==2:
			X = Data[:, :-1]; Y = Data[:, -1]; SeqSize = args.SeqStartPoint; 
			while(len(np.unique(Data[:args.SeqStartPoint, -1]))==1 and args.SeqStartPoint <=args.Budget): # make sure the training set include 2 classes
				args.SeqStartPoint+=1 
			if args.SeqStartPoint< args.Budget:
				if args.qs == 'Passive': # Passive learning or one-time training test
					start = time.time()

					Cls = GetClassifier(args, args.cls, Data[:args.SeqStartPoint])
					QueryIndex, TestingInd, DiscardforTrainInd, StopTime, Stats, PerProb = SequentialPRTwoSample(args, Cls, X, Y)
					end = time.time()				
				elif args.qs[:9] == 'V10Update': # classifier updating test; trained with adaptively collected samples 
					start = time.time(); 
	
					Cls = GetClassifier(args, args.cls, Data[:args.SeqStartPoint])

					QueryIndex, Stats = ClsUpdateV10SequentialPRTwoSample(args, Cls, X, Y)
					end = time.time()			
				
				
			if args.SaveStats == 1:
				QueryIndex = np.int16(QueryIndex);
				TestingInd = np.int16(TestingInd);
				DiscardforTrainInd = np.int16(DiscardforTrainInd);
				print('========================= Saving query index and prediction probability ==============================')
				with open(QueryPath + 'QueryIndex%d.txt'%(args.Trial), 'wb') as rp:
					pickle.dump(QueryIndex, rp);
				with open(QueryPath + 'TestingInd%d.txt'%(args.Trial), 'wb') as rp:
					pickle.dump(TestingInd, rp);
				with open(QueryPath + 'DiscardforTrainInd%d.txt'%(args.Trial), 'wb') as rp:
					pickle.dump(DiscardforTrainInd, rp)
				with open(QueryPath + 'PerProb%d.txt'%(args.Trial), 'wb') as rp:
					pickle.dump(PerProb, rp)		

		for u, p in enumerate(Per):
		
			print('==========  %.2f proportion, Sequential starting point:%d, Reject:%d, Stopping time:%d, NumTesting: %d, NumDisTrain: %d, NumDisFair: %d, trial %d, %.2fs elapsed =========='%
				(p, args.SeqStartPoint, Stats[1, u], Stats[0, u], Stats[2, u], Stats[3, u], Stats[4, u], args.Trial, end -start))

		np.save(StatsPath + 'StatsTrial%d.npy'%args.Trial, Stats)


def GetSPR(args, OnlinePredProb, AllPredProb, AllLambdaPredProb, AllY, QIndex, X, Y):

	if Y[QIndex] == 1:
		PredProb = OnlinePredProb[QIndex, 1]
	else:
		PredProb = OnlinePredProb[QIndex, 0]
	PredProb = max(PredProb, 1e-10)
	# print(PredProb) # debug
	LambdaPredProb = (1 - args.Lambda) * 0.5 + args.Lambda * PredProb # lambda-PUD


	AllPredProb.append(PredProb); 
	AllLambdaPredProb.append(LambdaPredProb); 

	AllY.append(Y[QIndex])
	P_Y1 = np.sum(AllY) / len(AllY); # estiamted label 1 probability 
	P_Y = np.array((1- P_Y1, P_Y1))
	All_P_Y = P_Y[np.int16(AllY)]; 
	All_P_Y_Ary = np.array(All_P_Y); AllPredProbAry = np.array(AllPredProb)
	AllLambdaPredProbAry = np.array(AllLambdaPredProb)

	# LogRatioSum = np.sum(np.log2(All_P_Y_Ary / AllPredProbAry))
	LambdaLogRatioSum = np.sum(np.log2(All_P_Y_Ary / AllLambdaPredProbAry))

	return LambdaLogRatioSum, PredProb


def GetSPR2(args, LambdaPredProb, AllLambdaPredProb, AllY, Y):
	Reject = 0;
	AllLambdaPredProb.append(LambdaPredProb); 
	AllY.append(Y)
	P_Y1 = np.sum(AllY) / len(AllY); # estiamted label 1 probability 
	P_Y = np.array((1- P_Y1, P_Y1))
	All_P_Y = P_Y[np.int16(AllY)]; 
	All_P_Y_Ary = np.array(All_P_Y); 
	AllLambdaPredProbAry = np.array(AllLambdaPredProb)

	TestN = len(All_P_Y_Ary)
	LambdaLogRatioSum = np.sum(np.log2(All_P_Y_Ary / AllLambdaPredProbAry))
	LambdaLogRatio = LambdaLogRatioSum/TestN; # Statisic scaled by log and sample size
	LogAlpha = np.log2(args.alpha) / TestN; # Alpha scaled by log and sample size

	# print(LogRatio, LogAlpha) # debug
	# print(LogRatio) #debug
	if LambdaLogRatio < LogAlpha:	
		Reject = 1; 
	return Reject, LambdaLogRatio


def ClsUpdateSequentialPRTwoSample(args, Cls, X, Y, StatsPath = None, QueryPath= None):
	P = 1; LogRatioSum = 0; n = args.SeqStartPoint; reject= 0; AllPredProbList = [];
	LambdaLogRatioSum = 0; AllLambdaPredProbList = []; # lambda pud 
	AllY = []
	Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);

	Stats = np.zeros((7, len(Per))); # The first row stores reject or accept and 
   								     # the second row stores stopping time
   								     # the third row stores the number testing points
   								     # the forth row stores the number of discarding examples for classifier training
   								     # the fifth row stores the number of discarding examples for fair prob
   								     # the sixth row stores the mutual information
   								     # the seventh row stores the statistic

	QueryIndex = np.arange(args.SeqStartPoint).tolist() # index of queried label
	TestingInd = [] # index of testing point
	DiscardforTrainInd = np.arange(args.SeqStartPoint).tolist() # index of discarding examples for classifier training
	UnqueryIndex = np.arange(args.SeqStartPoint, args.Budget).tolist() # index of unqueried labels
	TrainingIndex = np.arange(args.SeqStartPoint, args.Budget).tolist() # index of features waiting to be sampled for training
	STrainingIndex = np.arange(args.SeqStartPoint).tolist() # index of sampled features for training
	PerDiscardforFair = np.zeros(args.Budget); PerTestN = np.zeros(args.Budget); PerDiscardforTrain = np.zeros(args.Budget);
	PerMI = np.zeros(args.Budget); PerStats = np.zeros(args.Budget); PerProb = np.zeros(args.Budget)
	PerDiscardforTrain[:args.SeqStartPoint] = np.arange(1, args.SeqStartPoint + 1)
	DiscardforFair = 0; TestN = 0; DiscardforTrain = args.SeqStartPoint
	OnlinePredProb = -np.ones((args.Budget, 2)) # store allprediction probabilities online
	InitsizeList = [50, 100, 150, 200, 250, 300]; # candidate of training stop time for UAI method
	Initsize = random.choice(InitsizeList) 
	if args.qs[:8] == 'V6Update':
		# load statistic for V6 two-sample testing
		QueryIndex, PerProb, Stats = LoadV1Stat(args, StatsPath, QueryPath)
	else:
		while(len(QueryIndex) < args.Budget):

			AllPredProb = Cls.predict_proba(X[UnqueryIndex].reshape((-1, args.FeatLen)));
			OnlinePredProb[UnqueryIndex] = AllPredProb
		
			if args.qs == 'UpdateTrain_EnhanceUncertainty2' or args.qs == 'UAIUpdateTrain_EnhanceUncertainty2':
				
				AllPredProb0 = AllPredProb[:, 0];
				Index0 = np.argmax(AllPredProb0); Index1 = np.argmin(AllPredProb0); TwoIndex = [Index0, Index1]
				MaxMinProb0 =  [AllPredProb[Index0][0], AllPredProb[Index1][0]]; # maximum and minimum class 0 prob.
				if args.debias == 0:
					EstimatedP0 = args.prior
				elif args.debias == 1 or args.debias == 2:
					EstimatedP0 = np.sum(Y[np.int16(QueryIndex)] == 0) / (len(QueryIndex))
				PX0 = (0.5 - MaxMinProb0[1]) / max((MaxMinProb0[0] - MaxMinProb0[1]),1e-5); # marginal probability at x0
				PX1 = (MaxMinProb0[0] - 0.5) / max((MaxMinProb0[0] - MaxMinProb0[1]),1e-5); # marginal probability at x1
				# print(MaxMinProb0, PX0, PX1) # debug
				if PX0 < 0 or PX1 < 0 or PX0==PX1:
					Index = TwoIndex[np.random.choice(2, p=[0.5, 0.5])] # the probalities are not well learned and 
																		# both P(Y=0|x0) and P(Y=0 |x1) are larger 
																		# than zero.
				else:
					Index = TwoIndex[np.random.choice(2, p=[PX0, PX1])]
			elif args.qs == 'UpdateTrain_EnhanceUncertainty' or args.qs == 'UAIUpdateTrain_EnhanceUncertainty':
				AllPredProbMax = np.max(AllPredProb, 1); 
				Index = np.argmax(AllPredProbMax)
			QueryIndex.append(UnqueryIndex.pop(np.int16(Index))) # add query index
			TestingInd.append(QueryIndex[-1]) # add testing index
			TestN+=1 # add 1 to the number of testing points
			PerTestN[len(QueryIndex) - 1] = TestN
			PerDiscardforTrain[len(QueryIndex) - 1] = DiscardforTrain;
			# LogRatioSum, LambdaLogRatioSum, PredProb = GetSPR(args, OnlinePredProb, LogRatioSum, LambdaLogRatioSum, 
			# 												  AllPredProbList, AllLambdaPredProbList, 
			# 					 							  AllY, QueryIndex, X, Y) # get probability ratio

			LambdaLogRatioSum, PredProb = GetSPR(args, OnlinePredProb, AllPredProbList, 
												AllLambdaPredProbList, AllY, QueryIndex[-1], X, Y)
			# AllPredProb = np.delete(AllPredProb, Index, 0)
			PerProb[len(QueryIndex) - 1] = PredProb
			# LogRatio = LogRatioSum/TestN; # mutual information * (-1)
			LambdaLogRatio = LambdaLogRatioSum/TestN; # Statisic scaled by log and sample size
			LogAlpha = np.log2(args.alpha) / TestN; # Alpha scaled by log and sample size
			# PerMI[len(QueryIndex) - 1] = -LogRatio # mutual information 
			PerStats[len(QueryIndex) - 1] = -LambdaLogRatio # mutual information 
			# print(LogRatio, LogAlpha) # debug
			# print(LogRatio) #debug
			if LambdaLogRatio < LogAlpha:
				PerDiscardforTrain[len(QueryIndex) -1 :] = PerDiscardforTrain[len(QueryIndex) -1]
				PerTestN[len(QueryIndex) -1 :] = PerTestN[len(QueryIndex) -1]	
				# PerMI[len(QueryIndex) -1 :] = PerMI[len(QueryIndex) -1]		
				PerStats[len(QueryIndex) -1 :] = PerStats[len(QueryIndex) -1]		
				reject = 1; break
			if len(TrainingIndex) < Initsize or args.qs[9] != 'UAIUpdate':
				Index2 = random.choice(np.arange(len(TrainingIndex)))
				STrainingIndex.append(TrainingIndex[Index2]) # add index to training set
				if not (TrainingIndex[Index2] in QueryIndex): # this sample is not queried before
					if Y[TrainingIndex[Index2]] == 1:
						PredProb = OnlinePredProb[TrainingIndex[Index2], 1]
					else:
						PredProb = OnlinePredProb[TrainingIndex[Index2], 0]
					PredProb = max(PredProb, 1e-10)

					DiscardforTrain+=1 ; # discarding examples for training
					QueryIndex.append(TrainingIndex[Index2]) # add query index if the uniformly sampled feature is not labeled
					DiscardforTrainInd.append(TrainingIndex[Index2]) # add discarding example
					PerTestN[len(QueryIndex) - 1] = TestN
					# PerMI[len(QueryIndex) -1] = PerMI[len(QueryIndex) -2]	
					PerStats[len(QueryIndex) -1] = PerStats[len(QueryIndex) -2]	
					PerProb[len(QueryIndex) -1] = PredProb
					PerDiscardforTrain[len(QueryIndex) - 1] = DiscardforTrain;
					UnqueryIndex.remove(np.int16(TrainingIndex[Index2])) # delete a sample from the pool of unqueried examples

				TrainingIndex.pop(Index2)

				TrX = X[STrainingIndex]; TrY = Y[STrainingIndex]
				TrData = np.hstack((TrX, TrY.reshape(-1, 1)));
				# BoostData = OverSample(TrData) # handling imbalance data
				if args.cls == 'BAknn': # bayesian average knn
					Cls = UpdateBayesCls(args, Cls, TrData)
				elif args.cls == 'knn': # knn classifiers
					Cls.n_neighbors=math.ceil(len(TrX)**(args.gr)) 
					Cls.fit(TrX, TrY)
				else: # other classifiers 
					Cls.fit(TrX, TrY)
		
	if args.qs[:8] != 'V6Update':
		StopTime = len(QueryIndex);
		# pdb.set_trace()
		for u, p in enumerate(Per):
			MaxQ = np.int16(p * args.Budget) # number of maximum queried labels 
			if MaxQ<StopTime:
				Stats[0, u] = MaxQ; Stats[1, u] = 0
			elif MaxQ > StopTime:
				Stats[0, u] = StopTime; Stats[1, u] = 1
			elif MaxQ == StopTime:
				Stats[0, u] = StopTime; Stats[1, u] = reject
			Stats[2, u] = PerTestN[MaxQ - 1]	
			Stats[3, u] = PerDiscardforTrain[MaxQ - 1]
			Stats[4, u] = PerDiscardforFair[MaxQ - 1]
			# Stats[5, u] = PerMI[MaxQ - 1]
			Stats[6, u] = PerStats[MaxQ - 1]
	else:
		StopTime = len(QueryIndex);
		for u, p in enumerate(Per):
			MaxQ = np.int16(p * args.Budget) # number of maximum queried labels 
			if MaxQ<StopTime:
				Stats[0, u] = MaxQ; 
				Stats[1, u] = GetSPR_IncludeUniform(args, PerProb, Y, QueryIndex, MaxQ)
			elif MaxQ > StopTime:
				Stats[0, u] = StopTime; Stats[1, u] = 1
			elif MaxQ == StopTime:
				Stats[0, u] = StopTime; 
				Stats[1, u] = reject
			Stats[2, u] = PerTestN[MaxQ - 1]	
			Stats[3, u] = PerDiscardforTrain[MaxQ - 1]
			Stats[4, u] = PerDiscardforFair[MaxQ - 1]
			# Stats[5, u] = PerMI[MaxQ - 1]
			Stats[6, u] = PerStats[MaxQ - 1]
	return QueryIndex, TestingInd, DiscardforTrainInd, StopTime, Stats, PerProb


def ClsUpdateV10SequentialPRTwoSample(args, Cls, X, Y, StatsPath = None, QueryPath= None):
	P = 1; LogRatioSum = 0; n = args.SeqStartPoint; reject= 0; AllPredProbList = [];
	LambdaLogRatioSum = 0; AllLambdaPredProbList = []; # lambda pud 
	AllY = []
	Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);

	Stats = np.zeros((7, len(Per))); # The first row stores reject or accept and 
   								     # the second row stores stopping time
   								     # the third row stores the number testing points
   								     # the forth row stores the number of discarding examples for classifier training
   								     # the fifth row stores the number of discarding examples for fair prob
   								     # the sixth row stores the mutual information
   								     # the seventh row stores the statistic

	QueryIndex = np.arange(args.SeqStartPoint).tolist() # index of queried label
	TestingInd = [] # index of testing point
	DiscardforTrainInd = np.arange(args.SeqStartPoint).tolist() # index of discarding examples for classifier training
	UnqueryIndex = np.arange(args.SeqStartPoint, args.Budget).tolist() # index of unqueried labels
	TrainingIndex = np.arange(args.SeqStartPoint, args.Budget).tolist() # index of features waiting to be sampled for training
	STrainingIndex = np.arange(args.SeqStartPoint).tolist() # index of sampled features for training
	PerDiscardforFair = np.zeros(args.Budget); PerTestN = np.zeros(args.Budget); PerDiscardforTrain = np.zeros(args.Budget);
	PerMI = np.zeros(args.Budget); PerStats = np.zeros(args.Budget); PerProb = np.zeros(args.Budget)
	PerDiscardforTrain[:args.SeqStartPoint] = np.arange(1, args.SeqStartPoint + 1)
	DiscardforFair = 0; TestN = 0; DiscardforTrain = args.SeqStartPoint
	OnlinePredProb = -np.ones((args.Budget, 2)) # store allprediction probabilities online


	while(len(QueryIndex) < args.Budget):

		AllPredProb = Cls.predict_proba(X[UnqueryIndex].reshape((-1, args.FeatLen)));
		OnlinePredProb[UnqueryIndex] = AllPredProb
	
		if args.qs == 'V10UpdateTrain_EnhanceUncertainty2':
			
			AllPredProb0 = AllPredProb[:, 0];
			Index0 = np.argmax(AllPredProb0); Index1 = np.argmin(AllPredProb0); TwoIndex = [Index0, Index1]
			MaxMinProb0 =  [AllPredProb[Index0][0], AllPredProb[Index1][0]]; # maximum and minimum class 0 prob.
			if args.debias == 0:
				EstimatedP0 = args.prior
			elif args.debias == 1 or args.debias == 2:
				EstimatedP0 = np.sum(Y[np.int16(QueryIndex)] == 0) / (len(QueryIndex))
			PX0 = (0.5 - MaxMinProb0[1]) / max((MaxMinProb0[0] - MaxMinProb0[1]),1e-5); # marginal probability at x0
			PX1 = (MaxMinProb0[0] - 0.5) / max((MaxMinProb0[0] - MaxMinProb0[1]),1e-5); # marginal probability at x1
			# print(MaxMinProb0, PX0, PX1) # debug
			if PX0 < 0 or PX1 < 0 or PX0==PX1:
				Index = TwoIndex[np.random.choice(2, p=[0.5, 0.5])] # the probalities are not well learned and 
																	# both P(Y=0|x0) and P(Y=0 |x1) are larger 
																	# than zero; it can be proved that bimodal query 
																	# is the best scheme to increase the mutual information between 
																	# sample features and sample labels
			else:
				Index = TwoIndex[np.random.choice(2, p=[PX0, PX1])]
		elif args.qs == 'V10UpdateTrain_EnhanceUncertainty':
			AllPredProbMax = np.max(AllPredProb, 1); 
			Index = np.argmax(AllPredProbMax)
		QueryIndex.append(UnqueryIndex.pop(np.int16(Index))) # add query index
		TestingInd.append(QueryIndex[-1]) # add testing index
		TestN+=1 # add 1 to the number of testing points
		PerTestN[len(QueryIndex) - 1] = TestN
		PerDiscardforTrain[len(QueryIndex) - 1] = DiscardforTrain;
		# LogRatioSum, LambdaLogRatioSum, PredProb = GetSPR(args, OnlinePredProb, LogRatioSum, LambdaLogRatioSum, 
		# 												  AllPredProbList, AllLambdaPredProbList, 
		# 					 							  AllY, QueryIndex, X, Y) # get probability ratio

		LambdaLogRatioSum, PredProb = GetSPR(args, OnlinePredProb, AllPredProbList, 
											AllLambdaPredProbList, AllY, QueryIndex[-1], X, Y)
		# AllPredProb = np.delete(AllPredProb, Index, 0)
		PerProb[len(QueryIndex) - 1] = PredProb
		# LogRatio = LogRatioSum/TestN; # mutual information * (-1)
		LambdaLogRatio = LambdaLogRatioSum/TestN; # Statisic scaled by log and sample size
		LogAlpha = np.log2(args.alpha) / TestN; # Alpha scaled by log and sample size
		# PerMI[len(QueryIndex) - 1] = -LogRatio # mutual information 
		PerStats[len(QueryIndex) - 1] = -LambdaLogRatio # mutual information 
		# print(LogRatio, LogAlpha) # debug
		# print(LogRatio) #debug
		if LambdaLogRatio < LogAlpha:
			PerDiscardforTrain[len(QueryIndex) -1 :] = PerDiscardforTrain[len(QueryIndex) -1]
			PerTestN[len(QueryIndex) -1 :] = PerTestN[len(QueryIndex) -1]	
			# PerMI[len(QueryIndex) -1 :] = PerMI[len(QueryIndex) -1]		
			PerStats[len(QueryIndex) -1 :] = PerStats[len(QueryIndex) -1]		
			reject = 1; break
		
		# trained with adaptive collecte samples 

		TrX = X[QueryIndex]; TrY = Y[QueryIndex]
		TrData = np.hstack((TrX, TrY.reshape(-1, 1)));
		# BoostData = OverSample(TrData) # handling imbalance data
		if args.cls == 'BAknn': # bayesian average knn
			Cls = UpdateBayesCls(args, Cls, TrData)
		elif args.cls == 'knn': # knn classifiers
			Cls.n_neighbors=math.ceil(len(TrX)**(args.gr)) 
			Cls.fit(TrX, TrY)
		else: # other classifiers 
			Cls.fit(TrX, TrY)
	

	StopTime = len(QueryIndex);
	# pdb.set_trace()
	for u, p in enumerate(Per):
		MaxQ = np.int16(p * args.Budget) # number of maximum queried labels 
		if MaxQ<StopTime:
			Stats[0, u] = MaxQ; Stats[1, u] = 0
		elif MaxQ > StopTime:
			Stats[0, u] = StopTime; Stats[1, u] = 1
		elif MaxQ == StopTime:
			Stats[0, u] = StopTime; Stats[1, u] = reject
		Stats[2, u] = PerTestN[MaxQ - 1]	
		Stats[3, u] = PerDiscardforTrain[MaxQ - 1]
		Stats[4, u] = PerDiscardforFair[MaxQ - 1]
		# Stats[5, u] = PerMI[MaxQ - 1]
		Stats[6, u] = PerStats[MaxQ - 1]

	return QueryIndex, Stats


def SequentialPRTwoSample(args, Cls, X, Y):
	P = 1; LogRatioSum = 0; reject= 0; AllPredProbList = [];
	LambdaLogRatioSum=0; AllLambdaPredProbList = []; # for lambda pud probabilities

	Per = np.arange(args.Interval, args.Per + args.Interval, args.Interval);
	Stats = np.zeros((9, len(Per))); # The first row stores reject or accept and 
   								     # the second row stores stopping time
   								     # the third row stores the number testing points
   								     # the forth row stores the number of discarding examples for classifier training
   								     # the fifth row stores the number of discarding examples for fair prob
   								     # the sixth row stores the mutual information
   								     # the seventh row stores the statistic
   								     # the eighth row stores the switching sign
   								     # the ninth row stores the brier score

	QueryIndex = np.arange(args.SeqStartPoint).tolist() # index of queried label
	TestingInd = [] # index of testing point
	DiscardforTrainInd = np.arange(args.SeqStartPoint).tolist() # index of discarding examples for classifier training
	UnqueryIndex = np.arange(args.SeqStartPoint, args.Budget).tolist() # index of unqueried labels
	STrainingIndex = np.arange(args.SeqStartPoint).tolist() # index of sampled features for training
	PerDiscardforFair = np.zeros(args.Budget); PerTestN = np.zeros(args.Budget); PerDiscardforTrain = np.zeros(args.Budget);
	PerDiscardforTrain[: args.SeqStartPoint] = np.arange(1, args.SeqStartPoint + 1);
	PerMI = np.zeros(args.Budget);PerStats = np.zeros(args.Budget);PerProb = np.zeros(args.Budget);
	PerSwitch = np.zeros(args.Budget);PerEva = np.zeros(args.Budget);
	DiscardforFair = 0; TestN = 0; DiscardforTrain = args.SeqStartPoint; 
	OnlinePredProb = -np.ones((args.Budget, 2)) # store allprediction probabilities online
	SwitchSign = 0; Eva = 0; SwitchTime=0

	while(len(QueryIndex)<args.Budget):
		# n=i + 1

		AllPredProb = Cls.predict_proba(X[UnqueryIndex].reshape((-1, args.FeatLen))); 
		OnlinePredProb[UnqueryIndex] = AllPredProb
		if SwitchSign==0 and args.qs!= 'Passive':
			SwitchSign, Eva = SwitchDetect(args, X[STrainingIndex], Y[STrainingIndex])
			# print(Eva, SwitchSign)
			if SwitchSign == 1:
				SwitchTime = len(QueryIndex) + 1
				print('Switch time:%d, :Eva:%.4f'%(SwitchTime,Eva))
		if args.qs == 'OneTimeTrain_EnhanceUncertainty2' and SwitchSign==1:
			AllPredProb0 = AllPredProb[:, 0];
			Index0 = np.argmax(AllPredProb0); Index1 = np.argmin(AllPredProb0); TwoIndex = [Index0, Index1]
			MaxMinProb0 =  [AllPredProb[Index0][0], AllPredProb[Index1][0]]; # maximum and minimum class 0 prob.
			PX0 = (0.5 - MaxMinProb0[1]) / max((MaxMinProb0[0] - MaxMinProb0[1]), 1e-5); # marginal probability at x0
			PX1 = (MaxMinProb0[0] - 0.5) / max((MaxMinProb0[0] - MaxMinProb0[1]), 1e-5); # marginal probability at x1
			# print(MaxMinProb0, PX0, PX1) # debug
			if PX0 < 0 or PX1 < 0:
				Index = TwoIndex[np.random.choice(2, p=[0.5, 0.5])] # the probalities are not well learned and 
																	# both P(Y=0|x0) and P(Y=0 |x1) are larger 
																	# than zero.
			else:
				Index = TwoIndex[np.random.choice(2, p=[PX0, PX1])]
			
		elif args.qs == 'OneTimeTrain_EnhanceUncertainty' and SwitchSign==1:
			AllPredProbMax = np.max(AllPredProb, 1); 
			Index = np.argmax(AllPredProbMax)
			
		elif args.qs == 'Passive' or SwitchSign==0:
			Index = np.random.choice(len(UnqueryIndex))
			STrainingIndex.append(UnqueryIndex[Index])

		QueryIndex.append(UnqueryIndex.pop(np.int16(Index))) # add query index	
		TestingInd.append(QueryIndex[-1])		
		TestN+=1 # add 1 to the number of testing points
		PerTestN[len(QueryIndex) - 1] = TestN
		PerDiscardforFair[len(QueryIndex) - 1] = DiscardforFair
		PerDiscardforTrain[len(QueryIndex) - 1] = DiscardforTrain;
		PerSwitch[len(QueryIndex) -1] = SwitchTime
		PerEva[len(QueryIndex)-1]= Eva;
		if Y[QueryIndex[-1]] == 1:
			PredProb = OnlinePredProb[QueryIndex[-1], 1]
		else:
			PredProb = OnlinePredProb[QueryIndex[-1], 0]
		PredProb = max(PredProb, 1e-10); PerProb[len(QueryIndex) - 1] = PredProb

		LambdaPredProb = (1 - args.Lambda) * 0.5 + args.Lambda * PredProb # lambda-PUD

		if args.debias == 1:
			if Y[QueryIndex[-1]] == 1:
				# P_Y = np.sum(Y[:Index] == 1) / Index; # estiamted label 1 probability 
				P_Y = np.sum(Y[QueryIndex[:-1]] == 1) / (len(QueryIndex) - 1); # estiamted label 1 probability 
			else:
				# P_Y = 1 - np.sum(Y[:i] == 1) / i # estiamted label 0 probability 
				P_Y = 1 - np.sum(Y[QueryIndex[:-1]] == 1) / (len(QueryIndex) - 1)
			LogRatioSum += np.log2(P_Y / PredProb)
			LambdaLogRatioSum += np.log2(P_Y / LambdaPredProb)
		elif args.debias == 2:
			# P_Y1 = np.sum(Y[:n] == 1) / n; # estiamted label 1 probability 
			P_Y1 = np.sum(Y[QueryIndex[args.SeqStartPoint:]] == 1) / len(QueryIndex[args.SeqStartPoint:]); # estiamted label 1 probability 
			P_Y = np.array((1- P_Y1, P_Y1))
			All_P_Y = P_Y[np.int16(Y[QueryIndex[args.SeqStartPoint:]])]; 
			# print(PredProb) #debug
			AllPredProbList.append(PredProb)
			AllLambdaPredProbList.append(LambdaPredProb)
			
			All_P_Y_Ary = np.array(All_P_Y); AllPredProb_Ary = np.array(AllPredProbList); 
			AllLambdaPredProb_Ary = np.array(AllLambdaPredProbList); 

			LambdaLogRatioSum = np.sum(np.log2(All_P_Y_Ary/AllLambdaPredProb_Ary))
			LogRatioSum = np.sum(np.log2(All_P_Y_Ary/AllPredProb_Ary))
			# P_H0 = np.cumprod(All_P_Y)[-1]; P_H1 = np.cumprod(AllPredProbList)[-1]
			# Ratio = P_H0 / P_H1
		elif args.debias == 0:
			if Y[QueryIndex[-1]] == 1:
				# P_Y = 1 - args.prior
				P_Y = 0.5
			else:
				# P_Y = args.prior
				P_Y = 0.5
			LogRatioSum +=np.log2(P_Y / PredProb)
			LambdaLogRatioSum +=np.log2(P_Y / LambdaPredProb)
			# Ratio = Ratio * (P_Y / PredProb)
			# print(np.log2(P_Y / PredProb), PredProb) #debug

		LogRatio = LogRatioSum / TestN; # statistic scaled by log and testing sample size
		LambdaLogRatio = LambdaLogRatioSum / TestN; # statistic scaled by log and testing sample size
		LogAlpha = np.log2(args.alpha) / TestN; # alpha scaled by log and testing sample size		
		# print(LogRatio, LogAlpha)	# debug
		PerMI[len(QueryIndex) -1] = -LogRatio; # mutual information
		PerStats[len(QueryIndex) -1] = -LambdaLogRatio; # statistics
		if LambdaLogRatio < LogAlpha:
			PerTestN[len(QueryIndex) -1 :] = PerTestN[len(QueryIndex) -1]
			PerMI[len(QueryIndex) -1 :] = PerMI[len(QueryIndex) -1]
			PerStats[len(QueryIndex) -1 :] = PerStats[len(QueryIndex) -1]
			PerSwitch[len(QueryIndex) -1:] = PerSwitch[len(QueryIndex) -1]
			PerEva[len(QueryIndex) -1:] = PerEva[len(QueryIndex) -1]
			reject=1; break
		TrX = X[STrainingIndex]; TrY = Y[STrainingIndex]
		TrData = np.hstack((TrX, TrY.reshape(-1, 1)));
		# BoostData = OverSample(TrData) # handle imbalance data
		# TrX = BoostData[:, :-1]; TrY = BoostData[:, -1]
		if args.qs == 'Passive' or SwitchSign==0:
			if args.cls == 'BAknn':
				Cls = UpdateBayesCls(args, Cls, TrData)
			elif args.cls == 'knn':
				Cls.n_neighbors=math.ceil(len(TrX)**(args.gr)); 
				Cls.fit(TrX, TrY)
			else:
				Cls.fit(TrX, TrY)
	StopTime = len(QueryIndex);
	# pdb.set_trace() # debug
	for u, p in enumerate(Per):
		if p * args.Budget <StopTime:
			Stats[0, u] = p * args.Budget; Stats[1, u] = 0
		elif p * args.Budget > StopTime:
			Stats[0, u] = StopTime; Stats[1, u] = 1
		elif p*args.Budget == StopTime:
			Stats[0, u] = StopTime; Stats[1, u] = reject
		Stats[2, u] = PerTestN[np.int16(p * args.Budget) - 1]	
		Stats[3, u] = PerDiscardforTrain[np.int16(p * args.Budget) - 1]
		Stats[4, u] = PerDiscardforFair[np.int16(p * args.Budget) - 1]
		Stats[5, u] = PerMI[np.int16(p * args.Budget) - 1]
		Stats[6, u] = PerStats[np.int16(p * args.Budget) - 1]
		Stats[7, u] = PerSwitch[np.int16(p * args.Budget) - 1]
		Stats[8, u] = PerEva[np.int16(p * args.Budget) - 1]
	return QueryIndex, TestingInd, DiscardforTrainInd, StopTime, Stats, PerProb	



def GetClassifier(args, ClsName, Trdata):
	"""
	Classifier for passive learning
	"""

	# BoostTrData = OverSample(Trdata) # imbalance handle
	# X = BoostTrData[:, :-1]; Y = BoostTrData[:, -1]
	X = Trdata[:, :-1]; Y = Trdata[:, -1]

	if ClsName=='logistic':
		Cls = LogisticRegression(random_state=args.Trial); 
	elif ClsName == 'SVC':
		Cls = SVC(gamma='auto', random_state=args.Trial, probability = True); 
	elif ClsName == 'knn' or ClsName == 'PCAKnn':
		Cls = KNeighborsClassifier(algorithm='auto', n_neighbors=math.ceil(len(Trdata)**(args.gr))); 
	elif ClsName =='NN':
		Cls = BuildNN(len(X)); 
		X = torch.tensor(X).float(); Y = torch.tensor(Y.reshape(-1)).long();
	elif ClsName == 'CaliKnn':
		Cls = KNeighborsClassifier(algorithm='auto', n_neighbors=math.ceil(len(Trdata)**(args.gr)));
		if len(Trdata) > 50:
			Cls = CalibratedClassifierCV(base_estimator=Cls, n_jobs=mp.cpu_count(), ensemble=False); 
	elif ClsName =='CaliNN':
		Cls = BuildNN(X.shape[1]); 
		if len(Trdata) > 50:
			Cls = CalibratedClassifierCV(base_estimator=Cls,n_jobs=mp.cpu_count(), ensemble=False) 
		X = torch.tensor(X).long(); Y = torch.tensor(Y.reshape(-1)).int();
	elif ClsName =='CaliSVC':
		Cls = SVC(random_state=args.Trial, gamma='auto', probability = True, class_weight= 'balanced')
		if len(Trdata) > 50 and np.sum(Y) >=2:
			Cls = CalibratedClassifierCV(base_estimator=Cls, n_jobs=mp.cpu_count(), ensemble=False); 
	Cls.fit(X,Y); 

	return Cls










