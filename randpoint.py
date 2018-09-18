# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:28:49 2018

@author: brand
"""
from PIL import Image
from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


for i in range(0,1000):

	plt.clf()

	center_1 = np.array([1,1])
	center_2 = np.array([1,1])

	data_1 = np.random.randn(200, 2) + center_1
	data_2 = np.random.randn(200,2) + center_2

	data = np.concatenate((data_1,data_2), axis = 0)

	plt.scatter(data[:,0], data[:,1], s=10)

	plt.axis('off')
	plt.savefig("ClusterR" + str(i) + ".png",  bbox_inches='tight')