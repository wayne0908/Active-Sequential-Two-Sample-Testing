import numpy as np
import os  
import pdb
import sys
import random
import scipy
from itertools import permutations
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from Utility import *

def MutualInformation(arg, StatsPath, X, Y, p, qs, cluster_num=16):
    """
    Mutual information calculation
    StatsPath: str. Path to save
    X: array. Features; Y: array. Labels
    p: proportion of used queried labels
    qs: str. query strategy
    """
    kmeans = KMeans(n_clusters=cluster_num, random_state=arg.Trial).fit(X); Label = kmeans.labels_
    N = len(X); I = 0
    for i in range(cluster_num):
        for j in range(2):
            p_xy = np.sum(Y[Label==i] == j)/N; p_x = np.sum(Label==i)/N; p_y = np.sum(Y==j)/N 
            # print(p_xy, p_x, p_y)
            if p_xy!=0:
                I+=p_xy * np.log(p_xy/(p_x * p_y))
    if os.path.isfile(StatsPath + '%s.xlsx'%qs):
        wb=load_workbook(StatsPath + '%s.xlsx'%qs);ws1=wb["MI"]
    else:
        wb = Workbook(); ws1 = wb.create_sheet("MI");
    Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    ws1.cell(row=arg.Trial, column=p, value=I)
    ws1['%s%d'%(Alphabet[p-1], arg.Trial + 1)] ='=AVERAGE(%s%d:%s%d)'%(Alphabet[p-1], 1, Alphabet[p-1], arg.Trial)
    wb.save(StatsPath + '%s.xlsx'%qs)
    return I 


def MutualInformation2(args, StatsPath, X, Y, p, qs):
    """
    Mutual information calculation
    StatsPath: str. Path to save
    X: array. Features; Y: array. Labels
    p: proportion of used queried labels
    qs: str. query strategy
    """
    Mean0 = np.zeros(args.FeatLen); Mean1 = np.zeros(args.FeatLen);
    Mean0[0] = Mean0[0] - args.Sep / 2; Mean1[0] = Mean1[0] + args.Sep / 2 
    Cov0 = np.diag(np.ones(args.FeatLen)); Cov1 = np.diag(np.ones(args.FeatLen) +args.Del)
    Y0 = np.sum(Y == 0); Y1 = np.sum(Y==1); 
    R0 = Y0/(Y0 + Y1); R1 = Y0/(Y0 + Y1);

    P0 = multivariate_normal.pdf(X, mean = Mean0, cov =Cov0);
    P1 = multivariate_normal.pdf(X, mean = Mean1, cov =Cov1);    

    Pos0 = P0 * 0.5 / (P0 * 0.5 + P1 * 0.5)
    Pos1 = P1 * 0.5 / (P0 * 0.5 + P1 * 0.5)

    HY = -(R0 * np.log(R0) + R1 * np.log(R1)); HXY = -np.mean(Pos0*np.log(Pos0) + Pos1*np.log(Pos1))
    I = HY - HXY
    if os.path.isfile(StatsPath + '%s.xlsx'%qs):
        wb=load_workbook(StatsPath + '%s.xlsx'%qs);ws1=wb["MI"]
    else:
        wb = Workbook(); ws1 = wb.create_sheet("MI");
    Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    ws1.cell(row=args.Trial, column=p, value=I)
    ws1['%s%d'%(Alphabet[p-1], args.Trial + 1)] ='=AVERAGE(%s%d:%s%d)'%(Alphabet[p-1], 1, Alphabet[p-1], args.Trial)
    wb.save(StatsPath + '%s.xlsx'%qs)
    print(I)
    return I 

def NNError(args, StatsPath, X, p, qs):
    """
    Mutual information calculation
    StatsPath: str. Path to save
    X: array. Features;
    p: proportion of used queried labels
    qs: str. query strategy
    """
    Mean0 = np.zeros(args.FeatLen); Mean1 = np.zeros(args.FeatLen);
    Mean0[0] = Mean0[0] - args.Sep / 2; Mean1[0] = Mean1[0] + args.Sep / 2 
    Cov0 = np.diag(np.ones(args.FeatLen)); Cov1 = np.diag(np.ones(args.FeatLen) +args.Del)

    P0 = multivariate_normal.pdf(X, mean = Mean0, cov =Cov0);
    P1 = multivariate_normal.pdf(X, mean = Mean1, cov =Cov1);
   
    Pos0 = P0 * 0.5 / (P0 * 0.5 + P0 * 0.5)
    Pos1 = P1 * 0.5 / (P0 * 0.5 + P0 * 0.5)
    I = np.mean(Pos0 * Pos1)
    if os.path.isfile(StatsPath + '%s.xlsx'%qs):
        wb=load_workbook(StatsPath + '%s.xlsx'%qs);
        if not 'NNerr' in wb.sheetnames:
            ws1=wb.create_sheet("NNerr")
        else:
            ws1=wb["NNerr"]
    else:
        wb = Workbook(); ws1 = wb.create_sheet("NNerr");

    Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    ws1.cell(row=args.Trial, column=p, value=I)
    ws1['%s%d'%(Alphabet[p-1], args.Trial + 1)] ='=AVERAGE(%s%d:%s%d)'%(Alphabet[p-1], 1, Alphabet[p-1], args.Trial)
    wb.save(StatsPath + '%s.xlsx'%qs)
    return I 

def ClassifierEval(args, StatsPath, Score, X, p, qs):
    """
    Mutual information calculation
    StatsPath: str. Path to save
    X: array. Features;
    Score: class one posterial probability of classifier
    p: proportion of used queried labels
    qs: str. query strategy
    """
    Mean0 = np.zeros(args.FeatLen); Mean1 = np.zeros(args.FeatLen);
    Mean0[0] = Mean0[0] - args.Sep / 2; Mean1[0] = Mean1[0] + args.Sep / 2 
    Cov0 = np.diag(np.ones(args.FeatLen)); Cov1 = np.diag(np.ones(args.FeatLen) +args.Del)

    P0 = multivariate_normal.pdf(X, mean = Mean0, cov =Cov0);
    P1 = multivariate_normal.pdf(X, mean = Mean1, cov =Cov1);
   
    Pos0 = P0 * 0.5 / (P0 * 0.5 + P0 * 0.5)
    Pos1 = P1 * 0.5 / (P0 * 0.5 + P0 * 0.5)
    I = np.corrcoef(Score, Pos1)[0, 1]; 

    if os.path.isfile(StatsPath + '%s.xlsx'%qs):
        wb=load_workbook(StatsPath + '%s.xlsx'%qs);
        if not 'corr' in wb.sheetnames:
            ws1=wb.create_sheet("corr")
        else:
            ws1=wb["corr"]
    else:
        wb = Workbook(); ws1 = wb.create_sheet("corr");
    Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    ws1.cell(row=args.Trial, column=p, value=I)
    ws1['%s%d'%(Alphabet[p-1], args.Trial + 1)] ='=AVERAGE(%s%d:%s%d)'%(Alphabet[p-1], 1, Alphabet[p-1], args.Trial)
    wb.save(StatsPath + '%s.xlsx'%qs)
    return I 

def GetBER(args, Mean):
    """
    numerically compute BER
    Mean: center of the second component
    """
    X = np.zeros(args.FeatLen); X[0] = 0; X[1:] = np.inf
    BER = multivariate_normal.cdf(X, mean = Mean)
    return BER

def GetData(Args):
    """
    Args: parse(). option parameters
    Base:int. Base case or nonbase case
    """   
    if Args.DataType == 'Syn':
        Args.S = 2000
        Mean0 = np.zeros(Args.FeatLen); Mean1 = np.zeros(Args.FeatLen);
        Mean0[0] = Mean0[0] - Args.Sep / 2; Mean1[0] = Mean1[0] + Args.Sep / 2 
        Cov0 = np.diag(np.ones(Args.FeatLen)); Cov1 = np.diag(np.ones(Args.FeatLen) +Args.Del)
        BER = GetBER(Args, Mean1) # numerically compute BER
        print('Creating synthetic dataset of two component gaussian. Sample Size: %d, Prior:%.1f, Speration:%.2f, Delta:%2f, Dimension: %d, BER:%.2f'%(Args.S, Args.prior, Args.Sep, Args.Del, Args.FeatLen, BER))
        StatsPath = os.getcwd() + '/Stats/%s/Data/Prior%.1f/D%d/Sep%.2f/Delta%.2f/Size%d/'%(Args.DataType, Args.prior, Args.FeatLen, Args.Sep, Args.Del, Args.S)
       
        
        S0 = np.random.binomial(Args.S, Args.prior); S1 = Args.S - S0
        mn0 = multivariate_normal(Mean0, Cov0); 
        mn1 = multivariate_normal(Mean1,Cov1);
        trmn0 = multivariate_normal(Mean0, Cov0); 
        trmn1 = multivariate_normal(Mean1, Cov1);
        TrFeat0 = mn0.rvs(size=S0, random_state=Args.Trial); TrFeat1 = mn1.rvs(size=S1 , random_state=Args.Trial + 1)
        TeFeat0 = trmn0.rvs(size=S0, random_state=Args.Trial+2); TeFeat1 = trmn1.rvs(size=S1, random_state=Args.Trial+3)

        TrData0 = np.concatenate((TrFeat0, np.zeros((S0, 1))), 1); TeData0 = np.concatenate((TeFeat0, np.zeros((S0, 1))), 1)
        TrData1 = np.concatenate((TrFeat1, np.ones((S1, 1))), 1); TeData1 = np.concatenate((TeFeat1, np.ones((S1, 1))), 1)
  
        TrData = np.concatenate((TrData0, TrData1), 0); TeData = np.concatenate((TeData0, TeData1), 0);
        TrData = np.random.RandomState(Args.Trial-1).permutation(TrData); 
        TeData = np.random.RandomState(Args.Trial-1).permutation(TeData)

        if not os.path.exists(StatsPath):
            os.makedirs(StatsPath)
        if Args.SaveData:
            # np.save(StatsPath + 'SynSep%.2fDel%.2fSize%d.npy'%(Args.Sep, Args.Del, Args.S), Data); 
            # np.save(StatsPath + 'TrSynSep%.2fDel%.2fSize%d.npy'%(Args.Sep, Args.Del, Args.S), TrData)
            DrawData(Args, TrData, BER)
     

    elif Args.DataType == 'MNIST':
        Args.S = 2000
        print('loading %s dataset'%Args.DataType)
        StatsPath = os.getcwd() + '/Stats/%s/Data/Feat1/'%(Args.DataType);
        TwoDigitId = random.sample(list(np.arange(10)), 2); TwoR = random.sample(list(np.arange(10)), 4)        
      
        SampleX1 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], TwoR[0])); 
        SampleX2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], TwoR[1])); 
        TrSampleX = np.concatenate((SampleX1, SampleX2), 0); 
       
        SampleY1 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], TwoR[2]));
        SampleY2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], TwoR[3]));
        TrSampleY = np.concatenate((SampleY1, SampleY2), 0)

        ExSampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[1], TwoR[0])); QueryIndex = np.random.RandomState(Args.Trial-1).permutation(len(ExSampleY))
        TrSampleY[np.int64(QueryIndex[:600])] = ExSampleY[np.int64(QueryIndex[:600])];      
        TrSampleX[:, -1] = 0; TrSampleY[:, -1] = 1;Args.FeatLen=TrSampleX.shape[1] -1

        S0 = np.random.binomial(Args.S, Args.prior); S1 = Args.S - S0
        Index0 = np.random.choice(len(TrSampleX), S0, replace = False)
        Index1 = np.random.choice(len(TrSampleY), S1, replace = False)

        TeSampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], (TwoR[0] + 2) % 10)); TeSampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[0], (TwoR[0] + 3) % 10));
        TeSampleY2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(TwoDigitId[1], (TwoR[0] + 1) % 10)); 
        TeSampleY[np.int64(QueryIndex[:300])] = TeSampleY2[np.int64(QueryIndex[:300])]
        TeSampleX[:, -1] = 0; TeSampleY[:, -1] = 1; 

        TrData = np.vstack((TrSampleX[Index0], TrSampleY[Index1])); TeData = np.vstack((TeSampleX, TeSampleY));

        TrData = np.random.RandomState(Args.Trial-1).permutation(TrData); 
        TeData = np.random.RandomState(Args.Trial-1).permutation(TeData)
    elif Args.DataType == 'MNISTNull':
        Args.S = 2000; 
        print('loading %s dataset'%Args.DataType)
        StatsPath = os.getcwd() + '/Stats/MNIST/Data/Feat1/';perm = list(permutations(np.arange(10),2))
       
        DigitId = random.randint(0, 9); TwoR = random.sample(list(np.arange(10)), 4)

        TrSampleX1 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[0]))); 
        TrSampleX2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[1])));
        TrSampleX = np.concatenate((TrSampleX1, TrSampleX2), 0); 

        TrSampleY1 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[2])));
        TrSampleY2 = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[3])));
        TrSampleY = np.concatenate((TrSampleY1, TrSampleY2), 0);

        TrSampleX[:, -1] = 0; TrSampleY[:, -1] = 1; Args.FeatLen=TrSampleX.shape[1] -1

        S0 = np.random.binomial(Args.S, Args.prior); S1 = Args.S - S0
        Index0 = np.random.choice(len(TrSampleX), S0, replace = False)
        Index1 = np.random.choice(len(TrSampleY), S1, replace = False)
        
        TeSampleX = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[0] + 1)%10)); TeSampleY = np.load(StatsPath + 'MNISTl%dR%d.npy'%(int(DigitId), int(TwoR[1] + 1)%10));
        TeSampleX[:, -1] = 0; TeSampleY[:, -1] = 1;
      
        TrData = np.vstack((TrSampleX[Index0], TrSampleY[Index1]));TeData = np.vstack((TeSampleX, TeSampleY));
        TrData = np.random.RandomState(Args.Trial-1).permutation(TrData); TeData = np.random.RandomState(Args.Trial-1).permutation(TeData)        
     
    elif Args.DataType == 'ADNI':
        print('loading %s dataset'%Args.DataType)

        Args.S = 1000
        StatsPath = os.getcwd() + '/Stats/%s/Data/'%(Args.DataType); Data = np.load(StatsPath + 'NomalizedData.npy')
        Data0 = Data[Data[:, -1] == 0]; Data1 = Data[Data[:, -1] == 1]; Args.FeatLen=Data0.shape[1] -1
        S0 = np.random.binomial(Args.S, Args.prior); S1 = Args.S - S0
        AllIndex0 = np.arange(len(Data0)); AllIndex1 = np.arange(len(Data1))
        Index0 = np.random.choice(len(Data0), S0, replace = False)
        Index1 = np.random.choice(len(Data1), S1, replace = False)
        TeIndex0 = np.delete(AllIndex0, Index0); TeIndex1 = np.delete(AllIndex1, Index1);

        TrSample0 = Data0[Index0]; TrSample1 = Data1[Index1]; TrData = np.vstack((TrSample0, TrSample1)) 
        TeSample0 = Data0[TeIndex0]; TeSample1 = Data1[TeIndex1]; TeData = np.vstack((TeSample0, TeSample1)) 
        TrData = np.random.RandomState(Args.Trial-1).permutation(TrData); 
        TeData = np.random.RandomState(Args.Trial-1).permutation(TeData)
    if Args.cls == 'PCAKnn':
        TrX = TrData[:, :-1]; TrY = TrData[:, -1];
        TeX = TeData[:, :-1]; TeY = TeData[:, -1];
        # pca = PCA(n_components=2);
        # pca = KernelPCA(n_components=4, kernel = 'rbf');
        pca = KernelPCA(n_components=6, kernel = 'rbf');
        PCTrX = pca.fit_transform(TrX); PCTeX = pca.fit_transform(TeX)
        TrData = np.hstack((PCTrX, TrY.reshape((-1,1)))); TeData = np.hstack((PCTeX, TeY.reshape((-1,1)))); 
        Args.FeatLen = PCTrX.shape[1]
    return TrData, TeData 