import numpy as np 
import os 
import pdb
import random
import pickle
import sys
import time
import matlab.engine
from Dataset import * 
sys.path.append(os.getcwd() + "/ClassifierTwoSample/")
from ClassifierTwoSampleStats import *
# from Utility import *

def CreateDirectory(args, StatsPath, FigurePath, QueryPath):
	QueryPath = QueryPath + 'unbias%d/%s/QueryClsSize%d/'%(args.debias, args.qs, args.InitSize)
	if args.qs == 'Passive':
		if args.TestType == 'Binomial' or args.TestType == 'Hoeffding':
			StatsPath = StatsPath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.InitSize)
			FigurePath = FigurePath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.InitSize)
		elif args.TestType == 'Sequential' or args.TestType == 'SequentialPT':
			StatsPath = StatsPath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.SeqStartPoint)
			FigurePath = FigurePath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.SeqStartPoint)	
	elif args.qs[:6] == 'Update':
		StatsPath = StatsPath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.SeqStartPoint)
		FigurePath = FigurePath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.SeqStartPoint)
	else:
		StatsPath = StatsPath + 'unbias%d/%s/%s/QueryClassifierSize%d/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.KFold, args.SeqStartPoint)
		# FigurePath = FigurePath + 'unbias%d/%s/%s/TwoSampleClsSize%d/'%(args.debias,args.qs, args.TestType, args.SeqStartPoint)		

	if not os.path.exists(QueryPath):
		os.makedirs(QueryPath)
	if not os.path.exists(StatsPath):
		os.makedirs(StatsPath)
	# if not os.path.exists(FigurePath):
	# 	os.makedirs(FigurePath)	
	return StatsPath, QueryPath
def ClassifierTwoSampleTest(args, Data, HoldoutData, StatsPath, FigurePath, QueryPath):
	if args.RunTest == 1:
		# Create directory
		print('========================= Performing a %s Classifier Two-sample Test with %s query scheme =============================='%(args.TestType, args.qs))
		StatsPath2, QueryPath2 = CreateDirectory(args, StatsPath, FigurePath, QueryPath)	
		SequentialPRT(args, StatsPath2, QueryPath2, Data, HoldoutData)
