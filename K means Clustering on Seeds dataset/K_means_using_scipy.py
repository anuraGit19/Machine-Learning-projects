import numpy as np
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt

class K_means:
	def __init__(self):
		pass

	def load_data(self, path):
		self.data = np.loadtxt(path)
		self.data = self.data[:,:-1]
		
	def __calc_distance__(self,cluster_centers, labels):
		dist= 0 #np.zeros((len(cluster_centers), len(cluster_centers)))
		for j in range(len(self.data)):
			obj= (cluster_centers[labels[j]]-self.data[j])**2
			dist = dist + np.sum(obj)
		return dist	

	def k_means_algo_scikit(self, c):
		kmeans = KMeans(n_clusters=c, random_state=0).fit(self.data)
		return self.__calc_distance__(kmeans.cluster_centers_, kmeans.labels_)
	
	def k_means_algo(self, c):
		return self.__calc_distance__(kmeans.cluster_centers_)

if __name__ == "__main__":
	txtFilePath="."
	if len(sys.argv)>1:
		txtFilePath = sys.argv[1]
	km = K_means()
	km.load_data(txtFilePath)
	obj_function=[]
	clusters=[]
	k=[2,3,4,5,6,7,8,9,10]
	for i in k:
		values = km.k_means_algo_scikit(i)
		obj_function.append(values)

	plt.plot(k,obj_function,'b')
	plt.xlabel('Number of Clusters (k)')
	plt.ylabel('Objective function')
	plt.title('K-means plot')
	plt.legend(loc=4)
	plt.show()