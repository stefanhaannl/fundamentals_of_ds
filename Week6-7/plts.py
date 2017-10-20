import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
import random
import numpy as np

def plot_scatter(actual, predicted):
	plt.figure()
	plt.xlabel('Actual value')
	plt.ylabel('Predicted value')
	plt.title('Scatter plot predictions vs actual values')
	plt.scatter(actual, predicted, alpha=0.5)
	plt.show()

def plot_coefs(coefs, headernames):
	colors=[]
	for i in coefs:
		if i < 0:
			colors.append('red')
		else:
			colors.append('green')
			
	x = np.arange(len(headernames))
	
	plt.figure()
	plt.bar(x, coefs, align='center', color = colors, alpha = 0.8)
	plt.xticks(x, headernames)
	plt.xlabel('Feature')
	plt.ylabel('Coefficient value')
	plt.title('Feature importance')
	plt.show()
	
if __name__ == "__main__": 
	
	# plot scatter
	actual = [random.randint(0,100) for r in xrange(100)]	
	predicted = [random.randint(0,100) for r in xrange(100)]	
	plot_scatter(actual, predicted)
	
	# plot coefs	
	coefs = [1,2,-3,4]
	headernames = ['Bill', 'Fred', 'Mary', 'Sue']
	plot_coefs(coefs, headernames)
	
	
	

