A cluster Probability optimized Cost SensitiveSVM method.

Imbalanced classes present a challenging problem in data mining community
and real-world applications. In many situations, those minority examples are
of much more interest.We propose a cluster Probability optimized Cost SensitiveSVM 
method, called PCS-SVM and its results on various standard benchmark data 
sets and real-world data with different imbalance ratios show that the proposed 
method is effective compared with commonly used cost-sensitive techniques.

Note!!!
This codes is related to the Paper - 
Cluster Probability based Regularization Parameter Optimizing for Imbalanced SVM Classification

All files and illustrations：
SVM.py 
            --------------------- the implementation of the PCS-SVM model
Block_diagonal.py 
            ---------- a 'class' to calculate the similarity matrix between data        
hparams.py 
            ----------------- the file to set the parameters of the PCS-SVM      
tools.py 
            ------------------- the auxiliary tools to calculate some values of the PCS-SVM       
validate.py 
            ---------------- to run the PCS-SVM and get the results       
data 
            ----------------------- the document includes some datasets  
            
All the codes are written in Python and before you run the codes please 
make sure that you have installed all the needed software and library bolow:
1. Python 3.X
2. Numpy
3. sklearn 

By the way, you can also try to use Python2.X, I think it will work.
In our codes, we use only one method of sklearn to partition the datasets during
cross validation.In the experiments,for each data set, we processed a fivefold cross
validation and repeated the procession 20 times; the results are provided as mean values.

