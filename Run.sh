for t in {1..200}
do 
	for p in 0.5 0.6 0.7 0.8
	do 
		for u in 2
		do 
			python main.py --DataType='Syn' --Trial=$t --prior=$p --debias=$u --FeatLen=2 --Sep=0.4 --Del=0 --S=2000 --Budget=2000 --qs='Passive' --TestType='SequentialPT' --InitSize=50 --SeqStartPoint=10 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0  --Plot_AllStats=0  --Plot_Trend=0  --DrawLabel=0  --Plot_CI=0 --SaveStats=0 --gr=0.9
			python main.py --DataType='Syn' --Trial=$t --prior=$p --debias=$u --FeatLen=2 --Sep=0.4 --Del=0 --S=2000 --Budget=2000 --qs='V10UpdateTrain_EnhanceUncertainty2' --TestType='SequentialPT' --InitSize=50 --SeqStartPoint=10 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0  --Plot_AllStats=0  --Plot_Trend=0  --DrawLabel=0  --Plot_CI=0 --SaveStats=0 --gr=0.9
		done
	done
done

for t in {200..200}
do 
	for p in 0.5 0.6 0.7 0.8  
	do 
		for u in 2
		do 
			python main.py --DataType='Syn' --Trial=$t --prior=$p --debias=$u --FeatLen=2 --Sep=0.6 --Del=0 --S=2000 --Budget=2000 --qs='V10UpdateTrain_EnhanceUncertainty2' --TestType='SequentialPT' --InitSize=50 --SeqStartPoint=10 --Interval=0.1 --Per=1 --cls='logistic' --RunTest=1 --LoadQuery=0  --Plot_AllStats=0  --Plot_Trend=0  --DrawLabel=0  --Plot_CI=0 --SaveStats=0 --gr=0.9 --Plot_UnfairLEPriorTypeII=1 --Plot_samplecomplexity=1 
		done
	done
done