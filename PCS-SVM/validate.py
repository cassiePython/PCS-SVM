from __future__ import division, print_function
import csv, os, sys
import numpy as np
from SVM import SVM
import tools
from sklearn.model_selection import KFold
import hparams

filepath = os.path.dirname(os.path.abspath(__file__))

def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

#F-Measure G-means
def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == 1)
    FP = np.sum(y_hat[idx] != y[idx])
    idx = np.where(y_hat == -1)
    FN = np.sum(y_hat[idx] != y[idx])
    if TP == 0:
        return 0, 0
    Precision = float(TP/(TP+FP))
    Sensitivity = float(TP/(TP+FN))
    Specificity = float(TN/(TN+FP))
    F_measure = float(2 * Precision * Sensitivity / (Precision + Sensitivity))
    G_means = float(np.sqrt(Sensitivity * Specificity))
    return F_measure, G_means


def main(filename='data/iris-virginica.txt', alpha_C=1.0, kernel_type='linear', epsilon=0.001):
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)
    '''
    For each data set, we processed a fivefold cross validation
    and repeated the procession 20 times; the results are provided
    as mean values. 
    '''
    k = hparams.kfold #fivefold cross validation
    times = hparams.times #repeated the procession 20 times
    F_measure_svm, G_means_svm = 0, 0 #record the old svm
    F_measure, G_means = 0, 0 #record the new svm
    '''
    Change here:
    because of the srand partition of the dataset and unbalanced dataset,
    it may emerge that the divided part of the original dataset is unbalanced
    extremely.
    time: 2017/9/20
    '''
    #----------------------------------------------
    times_div = 0
    #----------------------------------------------
    for i in range(times):
        print ("the %d step(s) is started ......" %(i))
        f_measure_svm, g_means_svm = 0, 0 #record the sum value of old svm in every times
        f_measure_new, g_means_new = 0, 0 #record the sum value of new svm in every times
        '''
        Change here:
        make the value of 'shuffle' is False
        time: 2017/9/20
        '''
        #----------------------------------------------
        #get the training & testing data using cross validation method
        kf = KFold(n_splits=k,shuffle=True,random_state=i)
        #----------------------------------------------
        divide_count = 0
        divide_count_new = 0
        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            #print (test_data)
            #the data used to cal the alpha_k & alpha_0
            data_alpha, IR = tools.undersample(train_data.tolist())
            ##print (IR)
            # Split data
            X_train, y_train = train_data[:,0:-1], train_data[:,-1].astype(int)
            X_test, y_test = test_data[:,0:-1], test_data[:,-1].astype(int)
            X_alpha, y_alpha = data_alpha[:,0:-1], data_alpha[:,-1].astype(int)

            # Initialize model
            model = SVM(C=alpha_C, kernel_type=kernel_type, epsilon=epsilon)
           
            # Fit model
            support_vectors, iterations = model.fit(X_train, y_train)
            # Support vector count
            sv_count = support_vectors.shape[0]
            # Make prediction
            y_hat = model.predict(X_train)#to cal the P next
            y_hat_old = model.predict(X_test)#to eval the original svm
            # Calculate accuracy
            f_measure_svm_tmp, g_means_svm_tmp = calc_acc(y_test, y_hat_old)

            '''
            Erre occured here - change here:
            add the condition of the length of P
            to make sure that P is not empty
            time: 2017/9/17
            '''
            #----------------------------------------------
            P = tools.getSetP(y_train, y_hat)
            if (f_measure_svm_tmp == 0) or (len(P) == 0):
            #----------------------------------------------
                divide_count = divide_count + 1
                divide_count_new = divide_count_new + 1
                print ("f_measure_svm_tmp:",f_measure_svm_tmp)
                print ("len(P):",len(P))
                continue
            
            f_measure_svm = f_measure_svm + f_measure_svm_tmp
            g_means_svm = g_means_svm + g_means_svm_tmp
            
            #Get the set P which are classified as false negative
            '''
            delete the two lines of codes
            and move the condition upper
            time: 2017/9/17
            '''
            #-------------------------------------------------
            #P = tools.getSetP(y_train, y_hat)
            #assert len(P) > 0 #make sure that P is not null
            #-------------------------------------------------
            #I think it may be occurred that the P is empty, but actually this
            #phenomenon doesn't emerge, so I annotate the codes below and using
            #the assert len(P) > 0 instead.
            #if len(P) == 0:
            #    print ("XXXXXXXXXXXXXXXXXXXXXX")
            #    return
            #    divide_count = divide_count + 1
            #    continue
            #print("the set P which are classified as false negative: ",P)
            ##print("the number of the Points of P is ",len(P))

            #D is the distance between each sample in P and hyperplane
            D = tools.getDistanceD(P, X_train, model.w, model.b)
            ##print("the Distance D is: ", D)

            #A is the sample which has the maxmum in D
            A = tools.getMaxmumA(D)
            ##print("the Maxmum points in D is A: ", A)

            #Select a sample B which is classified as true positive and is nearest to A
            B = tools.getNearestB(A, X_train, y_train, y_hat)
            ##print("the nearest point to A and is true positive is B: ", B)

            #Get the similarity Matrix
            matrix = tools.getSimilarity(X_train, y_train)
            #print ("the similarity Matrix is: ", matrix)
            #Print the KAB
            KAB = matrix[A[0]][B[0]]
            ##print ("KAB is: ", KAB)

            #Select a sample C which is classified as true positive and is nearest to B
            C = tools.getNearestC(B, X_train, y_train, y_hat)
            ##print("the nearest point to B and is true positive is C: ", C)
            #Print the KBC
            KBC = matrix[B[0]][C[0]]
            ##print ("KBC is: ", KBC)

            #Get the similarity Matrix of the alpha data
            matrix_alpha = tools.getSimilarity(X_alpha, y_alpha)
            #Get PAB and PBC
            PAB, PBC = tools.getPabAndPbc(KAB, KBC, matrix_alpha, X_alpha, y_alpha)
            ##print("PAB is %f and PBC is %f: " %(PAB, PBC))

            #cal the lower bound of C+
            cplus_lower = tools.getLowerBoundOfCplus(KAB, PAB, PBC)
            #print("the lower bound of C+ is: ", cplus_lower)
            
            #cal the new svm using C+ between [cplus_lower,IR]
            cplus = cplus_lower
            step = hparams.step
            cplus_fg_svm_temp = []
            while cplus <= IR:
                #training the new SVM using cplus
                model_new = SVM(C=alpha_C, kernel_type=kernel_type, epsilon=epsilon)
               
                # Fit new model
                support_vectors, iterations = model_new.fit(X_train, y_train, True, cplus)

                # Make prediction
                y_hat_new = model_new.predict(X_train)
    
                # Calculate accuracy
                f_measure, g_means = calc_acc(y_train, y_hat_new)
                cplus_fg_svm_temp.append((cplus, f_measure, g_means, model_new))
                cplus = cplus + step
                
            if len(cplus_fg_svm_temp) == 0:
                divide_count_new = divide_count_new + 1
                continue
            
            ##print ("The optimal C+ by f_measure is: ",
            ##       cplus_fg_svm_temp[cplus_fg_svm_temp.index(
            ##           max(cplus_fg_svm_temp, key=lambda x:x[1])
            ##           )])
            #Get the new svm by f_measure
            #note: if you want to get the new svm by g-means ,you should use x[2] instead of x[1]
            model_new = cplus_fg_svm_temp[cplus_fg_svm_temp.index(
                         max(cplus_fg_svm_temp, key=lambda x:x[hparams.eval_type])
                         )][3]
            #print (svm_new)
            # Make prediction
            y_hat_new = model_new.predict(X_test)
            # Calculate accuracy
            f_measure_tmp, g_means_tmp = calc_acc(y_test, y_hat_new)
            assert f_measure_tmp !=0
            print("F-measure:\t%.3f and G-means:\t%.3f" % (f_measure_tmp, g_means_tmp))
            f_measure_new = f_measure_new + f_measure_tmp
            g_means_new = g_means_new + g_means_tmp
        '''
        Change here:
        because of the srand partition of the dataset and unbalanced dataset,
        it may emerge that the divided part of the original dataset is unbalanced
        extremely. If 'zero' occurs, we ignore it and run again.
        time: 2017/9/20
        '''
        #----------------------------------------------
        if (k - divide_count == 0) or (k - divide_count_new == 0):
            times_div += 1
            continue
        #----------------------------------------------
        F_measure_svm = F_measure_svm + f_measure_svm / (k - divide_count)
        G_means_svm = G_means_svm + g_means_svm / (k - divide_count)

        F_measure = F_measure + f_measure_new / (k - divide_count_new)
        G_means = G_means + g_means_new / (k - divide_count_new)

        #print("now total F-measure and G-means:\t%.3f and G-means:\t%.3f" % (F_measure, G_means))
        
    #cal the average value
    times -= times_div
    F_measure_svm, G_means_svm = F_measure_svm / times, G_means_svm / times
    F_measure, G_means = F_measure / times, G_means / times
    #print the average value of F-measure & G-means of the old svm
    #print ("The average value of F_measure_svm is: ",F_measure_svm)
    #print ("The average value of G_means_svm is: ",G_means_svm)
    #print the average value of F-measure & G-means of the new svm
    #print ("The average value of F-measure is: ",F_measure)
    #print ("The average value of G-means is: ",G_means)
    if hparams.eval_type == 1:
        print ("The average value of F-measure of old svm is: ",F_measure_svm)
        print ("The average value of F-measure of new svm is: ",F_measure)
    elif hparams.eval_type == 2:
        print ("The average value of G-means of old svm is: ",G_means_svm)
        print ("The average value of G-means of new svm is: ",G_means)
if __name__ == '__main__':
    main(hparams.dataset,hparams.alpha_c,hparams.kernel_type,hparams.epsilon)
