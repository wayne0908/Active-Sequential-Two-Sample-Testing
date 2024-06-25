import argparse 
import numpy as np 

parser = argparse.ArgumentParser(description='Hypothesis testing with active learning')

parser.add_argument('--DataType', type = str, default = 'Syn', help = 'Data type')

parser.add_argument('--TestType', type = str, default = 'Batch', help = 'Batch or Sequential testing')

parser.add_argument('--qs', type = str, default = 'EnhanceUncertainty', help = 'query strategy: EnhanceUncertainty or uncertainty_sampling' )

parser.add_argument('--cls', type = str, default = 'knn', help = 'Classifier type')

parser.add_argument('--CVMetric', type = str, default = 'BrierScore', help = 'cross validation metric')

parser.add_argument('--BFcls', type = str, default = 'logistic', help = 'Classifier type before model switch')

parser.add_argument('--Sep', type = float, default = 1.5, help = 'Seperation between means')

parser.add_argument('--Del', type = float, default = 0, help = 'difference between variance between means')

parser.add_argument('--Per', type = float, default = 1, help = 'Proportion of unlabelled pool')

parser.add_argument('--BP', type = float, default = 1, help = 'Budget proportion')

parser.add_argument('--Interval', type = float, default = 0.1, help = 'percentage increment')

parser.add_argument('--Lambda', type = float, default = 0.99, help = 'lambda-PUD distribution')

parser.add_argument('--beta', type = float, default = 0.01, help = 'decaying constant for keeping probability')

parser.add_argument('--EvaThres', type = float, default = 0.25, help = 'Threshold for the cross validation metric')

parser.add_argument('--gr', type = float, default = 0.9, help = 'nearest number or partition number grow rate')

parser.add_argument('--alpha', type = float, default = 0.05, help = 'significance level')

parser.add_argument('--prior', type = float, default = 0.5, help = 'Prior probability of class zero')

parser.add_argument('--debias', type = int, default = 0, help = 'Adjust prior probability to debias one')

parser.add_argument('--SamplingSize', type = int, default = 10000, help = 'Metropolis-Hastings sampling size')

parser.add_argument('--Sum', type = int, default = 1, help = 'sum/averge excel results')

parser.add_argument('--FeatLen', type = int, default = 2, help = 'feature length')

parser.add_argument('--SeqStartPoint', type = int, default = 1, help = 'Sequential testing starting point')

parser.add_argument('--Plot_time', type = int, default = 0, help = 'Plot time or not')

parser.add_argument('--groupsize', type = int, default = 100, help = 'Sequential testing starting point')

parser.add_argument('--InitSize', type = int, default = 15, help = 'Initial size to train a classifier')

parser.add_argument('--Budget', type = int, default = 2000, help = 'Maximum query complexity (label budget)') # It is only for synthetic data under the null

parser.add_argument('--LoadData', type = int, default = 0, help = 'Load existing data or not')

parser.add_argument('--LookFowardSteps', type = list, default = [1,2,3,4,5], help = 'number of steps for the bimodal query before the uniform query')

parser.add_argument('--LookFowardSteps2', type = list, default = [[3, 1],[2, 1],[1,1],[1,2],[1,3]], help = 'number of steps for the bimodal query before the uniform query')

parser.add_argument('--S', type = int, default = 500, help = 'Sample size')

parser.add_argument('--RunTest', type = int, default = 1, help = 'Run the classifier two-sample test or not')

parser.add_argument('--SaveData', type = int, default = 0, help = 'Save data or not')

parser.add_argument('--Trial', type = int, default = 1, help = 'Trial number')

parser.add_argument('--DrawLabel', type = int, default = 0, help = 'Draw queried sample or not')

parser.add_argument('--Plot_Stats', type = int, default = 0, help = 'Plot stats distribution or not')

parser.add_argument('--Plot_Dimension', type = int, default = 0, help = 'Plot dimension results or not')

parser.add_argument('--Plot_CI', type = int, default = 0, help = 'Plot confindence interval for Type I error or not')

parser.add_argument('--Plot_FairPriorPlot_CI', type = int, default = 0, help = 'Plot confindence interval for fair prior Type I error or not')

parser.add_argument('--Plot_Trend', type = int, default = 0, help = 'Plot all results trend or not')

parser.add_argument('--Plot_UnfairPriorTypeII', type = int, default = 0, help = 'Plot unfair prior Type II error and stopping time in figures')

parser.add_argument('--Plot_FairPriorTypeII', type = int, default = 0, help = 'Plot fair prior Type II error and stopping time in figures')

parser.add_argument('--Plot_UnfairLEPriorTypeII', type = int, default = 0, help = 'Plot fair prior label efficient Type II error and stopping time in figures')

parser.add_argument('--Plot_samplecomplexity', type = int, default = 0, help = 'Plot sample complexity')

parser.add_argument('--Excel_TypeII', type = int, default = 0, help = 'Plot Type II error and stopping time in excel')

parser.add_argument('--Plot_AllStats', type = int, default = 0, help = 'Plot all stats distribution or not')

parser.add_argument('--LoadQuery', type = int, default = 0, help = 'Load query index or not')

parser.add_argument('--Plot_ClsProb', type = int, default = 0, help = 'Plot classification probability or not')

parser.add_argument('--WriteToExcel', type = int, default = 1, help = 'Write stats to excel')

parser.add_argument('--loadV1Stat', type = int, default = 0, help = 'Whether to load V1 two-sample testing stats')

parser.add_argument('--Plot_Converge', type = int, default = 0, help = 'Plot convergence or not')

parser.add_argument('--detect', type = int, default = 0, help = 'detect if stats exist or not')

parser.add_argument('--ReFillStats', type = int, default = 0, help = 'Refill stats or not')

parser.add_argument('--ConSize', type = int, default = 100, help = 'Confidental sample size')

parser.add_argument('--SaveStats', type = int, default = 0, help = 'Save statistic or not')

parser.add_argument('--KFold', type = int, default = 50, help = 'Number of folds to average statistic')

