import cv2
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class BOVHelpers:
	def __init__(self, n_clusters = 20):
		self.n_clusters = n_clusters
		self.kmeans_obj = KMeans(n_clusters = n_clusters)
		self.kmeans_ret = None
		self.descriptor_vstack = None
		self.mega_histogram = None
		self.clf  =	SVC()

	def cluster(self):
		"""	
		cluster using KMeans algorithm, 
		"""
		self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

	def developVocabulary(self,n_images, descriptor_list, kmeans_ret = None):
		
		"""
		Each cluster denotes a particular visual word 
		Every image can be represeted as a combination of multiple 
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word 
		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images
		"""

		self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
		old_count = 0
		for i in range(n_images):
			l = len(descriptor_list[i])
			for j in range(l):
				if kmeans_ret is None:
					idx = self.kmeans_ret[old_count+j]
				else:
					idx = kmeans_ret[old_count+j]
				self.mega_histogram[i][idx] += 1
			old_count += l

	def standardize(self, std=None):
		"""
		
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.
		"""
		if std is None:
			self.scale = StandardScaler().fit(self.mega_histogram)
			self.mega_histogram = self.scale.transform(self.mega_histogram)
		else:
			self.mega_histogram = std.transform(self.mega_histogram)

	def formatND(self, l):
		"""	
		restructures list into vstack array of shape
		M samples x N features for sklearn
		"""
		vStack = np.array(l[0])
		for remaining in l[1:]:
			vStack = np.vstack((vStack, remaining))
		self.descriptor_vstack = vStack.copy()
		return vStack

	def train(self, train_labels):
		"""
		uses sklearn.svm.SVC classifier (SVM) 
		"""
		self.clf.fit(self.mega_histogram, train_labels)
		print("Training completed")

	def predict(self, iplist):
		predictions = self.clf.predict(iplist)
		return predictions