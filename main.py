import numpy as np 
import sys
import pdb
import os 
import random 
from Options import * 
sys.path.append(os.getcwd() + "/CreateDataset/")
sys.path.append(os.getcwd() + "/ClassifierTwoSample/")
from Dataset import * 
from ClassifierTwoSample import *
from Utility import *
def main():

	"""
	Set up parameters
	"""
	args = parser.parse_args()

	"""
	Set random seed 
	"""
	print('======================= Random number with seed %d ========================='%args.Trial)
	random.seed(args.Trial)
	np.random.seed(args.Trial)

	"""
	Build directory
	"""
	print('========================= Build directory ==============================')
	StatsPath, FigurePath, QueryPath, QueryFigurePath = GetDirectory(args)
	
	"""
	Acquire data
	"""
	print('======================= Acquiring data ==============================')
	TrData, HoldoutData = GetData(args)


	ClassifierTwoSampleTest(args, TrData, HoldoutData, StatsPath, FigurePath, QueryPath)
	

	"""
	plot fair prior Type I error confidence itnerval 
	"""
	if args.Plot_FairPriorPlot_CI == 1:
		PlotUnfairPriorLETypeI(args)

	"""
	Plot sample complexity needed 
	"""
	if args.Plot_samplecomplexity ==1:
		PlotFairPriorResultsSampleTrend(args)

	
	"""
	Plot label efficient unfair prior Type II error in figures
	"""
	if args.Plot_UnfairLEPriorTypeII == 1:
		PlotUnfairSWResultsComp(args)


if __name__ == '__main__':
	main()