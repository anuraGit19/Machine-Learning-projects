import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

faces= scipy.io.loadmat(os.getcwd()+'/faces.mat')

def cosineDistance(x,y):
	return 1-np.divide(np.dot(x,y),np.linalg.norm(x)*np.linalg.norm(y))

def get_majority_votes(l,B):
	ones=0 
	twos=0
	for i in range(len(l)):
		if B[l[i]]==1:
			ones=ones+1
		else:
			twos=twos+1
	return 1 if ones>twos else 2

def calculate_train_error(a,k,A,B):
	# A=X_train, B=y_train , k
	error=0.0
	for i in range(A.shape[0]):
		l=[]
		for j in range(k):
			l.append(a[i][j])
		label = get_majority_votes(l,B)
		if label!=B[i]:
			error=error+1
	return error/A.shape[0]

def calculate_test_error(a,k,A,B,C,D):
	# a= dismatrix,k, A= X_train, B=X_test, C=y_test, D=y_train
	error=0.0
	for i in range(B.shape[0]):
		l=[]
		for j in range(k):
			l.append(a[i][j])
		label = get_majority_votes(l,D)
		if label!=C[i]:
			error=error+1
	return error/B.shape[0]


def knn_algo_train(A,B,k):
	dis=[[i for i in range(A.shape[0])] for y in range(A.shape[0])]
	a=[[i for i in range(A.shape[0])] for y in range(A.shape[0])]

	for i in range(A.shape[0]):
		for j in range(A.shape[0]):
			if i==j:
				dis[i][j]=0
			else:
				dis[i][j]=cosineDistance(A[i],A[j])
	for i in range(A.shape[0]):
		a= np.argsort(dis,axis=1)
	trainerror = calculate_train_error(a,k,A,B)
	return trainerror

def knn_algo_predict(A,B,C,D,k):
	dis=dis=[[i for i in range(A.shape[0])] for y in range(B.shape[0])]
	a=[[i for i in range(A.shape[0])] for y in range(B.shape[0])]
	i=0
	j=0
	for i in range(B.shape[0]):
		for j in range(A.shape[0]):
			if i==j:
				dis[i][j]=0
			else:
				dis[i][j]=cosineDistance(B[i],A[j])
	for i in range(B.shape[0]):
		a= np.argsort(dis,axis=1)
	testerror = calculate_test_error(a,k,A,B,C,D)
	return testerror				

X_train = faces['traindata']
y_train = faces['trainlabels']
X_test = faces['testdata']
y_test = faces['testlabels']
trainerror=[]
testerror=[]
k=[1,10,20,30,40,50,60,70,80,90,100]

for i in range(len(k)):
	trainerror.append(knn_algo_train(X_train, y_train,k[i]))
	testerror.append(knn_algo_predict(X_train, X_test, y_test, y_train,k[i]))

plt.plot(k,trainerror,'g', label="Training Error")
plt.plot(k, testerror, 'b', label="Testing Error")
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error')
plt.title('KNN Error Rate on Faces Dataset')
plt.legend(loc=4)
plt.show()
