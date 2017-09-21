#parameters used in the SVM
max_iter = 1000 #max iterators
kernel_type = 'linear' #the type of kernel
alpha_c = 1.0 #parameter of the svm
epsilon = 0.001 #error of the svm

#the standard of evaluating the model
#the meaning of the value of eval_type
# ---  1 : F-measure ---
# ---  2 : G-means   ---
eval_type = 1

#the step of the new svm using C+ between [cplus_lower,IR]
step = 0.5

#parameters used int the rbf kernel
#np.exp( -self.G * np.sum((x1 - x2.T) ** 2))
#if you use different dataset with the RBF kernel,
#you should change the value of G before.
alpha_g = 1.0

#parameters used int the polynomial kernel
alpha_y = 1.0
alpha_r = 0
alpha_d = 2

#dataset
dataset = 'data/vehicle0.txt'

#validate
kfold = 5  ##fivefold cross validation
times = 20 #repeated the procession 20 times


