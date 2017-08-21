'''
MIT Licence
Author : Anurag Solanki
Dataset : breast-cancer-wisconsin - http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%25
logistic regression
'''
import numpy as np 
import pandas as pd

class logistic_regresor():
	'''
	Cleaning and preprocessing of data
	'''
	def dataPreprocessing(self):
		df = pd.read_csv('breast-cancer-wisconsin.csv', sep=',', header=None)
		self.nparray = df.as_matrix()
		self.nparray = self.nparray[:,1:]
		#np.random.shuffle(self.nparray)
		self.y_total = self.nparray[:,-1]
		self.X_total = np.delete(self.nparray, -1 ,1)
		self.y_total[self.y_total<3] = 0
		self.y_total[self.y_total>3] = 1
		self._splitData()
	

	def train(self):	
		self.w = np.random.rand(self.X_train.shape[1],1)
		self.b = np.random.rand(1)
		self.X_train = np.concatenate([self.X_train, np.ones((self.X_train.shape[0],1))], axis=1)
		self.X_test = np.concatenate([self.X_test, np.ones((self.X_test.shape[0],1))], axis=1)
		self.w = np.concatenate([self.w, self.b.reshape((self.b.shape[0],1))], axis=0)
		self.w= self._stochastic_gradient_descent (self.X_train,self.y_train)

	def test(self):
		self.predicted_labels = self._get_predictions(self.X_test)
		self.accuracy = self._calc_accuracy(self.predicted_labels)
		print "Accuracy : ",self.accuracy
		return self.accuracy

	def _splitData(self):
		splitRatio = 0.67
		self.X_train = self.X_total[:int(splitRatio*len(self.X_total))]
		self.X_test = self.X_total[int(splitRatio*len(self.X_total)):]
		self.y_train = self.y_total[:int(splitRatio*len(self.y_total))]
		self.y_test = self.y_total[int(splitRatio*len(self.y_total)):]

	def _get_predictions (self, x):
		y = np.dot(x, self.w) + self.b
		p = self._sigmoid(y)
		labels =[]
		for i in p:
			if i >= 0.5:
				labels = np.append(labels, 1)
			else:
				labels = np.append(labels, 0)
		return labels

	def _stochastic_gradient_descent (self, X_train,y_train):
		iters=1000  	# No. of epochs
		eta=1e-3    	# Leaning rate
		for i in range(iters):
			z = np.dot(self.X_train, self.w)
			predictions = self._sigmoid(z)
			error = self.y_train.reshape(self.y_train.shape[0],1) - predictions
			grad = np.dot(self.X_train.T,error)
			self.w += eta * grad
		
 			# error=self.y_train-self.X_train.dot(self.w)
 			# print "---1---",self.w.shape
			# rmse = np.sqrt(np.mean((self.y_train - np.dot(self.X_train,self.w)) ** 2))
			# print "---2---",self.w.shape
			# #print("Iteration",i," | RMSE :" ,rmse)
			# gradient = -2/self.X_train.shape[0]* np.dot(self.X_train.T,error)
			# print "---3---",self.w.shape
			# self.w=self.w-eta*gradient
		return self.w
	
	def _calc_accuracy(self, predicted_labels):
		count =0
		for i in range(len(self.y_test)):
			if self.y_test[i] == predicted_labels[i]:
				count=count+1
		return float(count) * 100 / self.y_test.shape[0]

	def _sigmoid(self, x):
		return 1. / (1. + np.exp(-x))

if __name__ == "__main__":
	lr = logistic_regresor()
	lr.dataPreprocessing()
	lr.train()
	lr.test()
    