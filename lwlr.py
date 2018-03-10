# Locally Weighted Linear Regression is very inefficient because Parameters are calculated again for each test case
# But, it should give good results after tuning the hyper-parameter tau

import csv
import math
import numpy

def converge(t):
	for i in t:
		if abs(i) > epsilon:
			return False
	return True

def stochastic_gradient_descent(w,theta):
	for _ in range(max_n):
		for i in range(len(X_s)):
			x = numpy.array(X_s[i])
			t = [0]*len(theta)
			for j in range(len(theta)):
				t[j] = alpha*w[i]*(Y_s[i]-numpy.dot(numpy.array(theta),x))*x[j]
			for j in range(len(theta)):
				theta[j] = theta[j] + t[j]
				# print(theta)
			# if converge(t):
			# 	return theta
	return theta

def get_data(name):
	data = []
	with open(name, 'r') as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			for i in range(len(row)):
				row[i] = float(row[i])
			data = data + [row]
	return data

def arrange_data(data):
	Xs = [[]]*len(data)
	Ys = [[]]*len(data)
	for i in range(len(data)):
		Xs[i] = data[i][:-1]+[1]
		Ys[i] = data[i][-1]
	return Xs,Ys

def weight(x_i,x):
	x_i = numpy.array(x_i)
	x = numpy.array(x)
	temp = x_i-x
	temp = numpy.dot(temp,temp)
	return math.exp(-1.0*temp/(2*tau*tau))

def get_weights(Xs,x):
	# return [1]*len(Xs) # Uncomment If you want standard Linear Regression
	weights = [0]*len(Xs)
	for i in range(len(weights)):
		weights[i] = weight(Xs[i],x)
	return weights

def get_parameters(w,n):
	theta = [0]*n
	theta = stochastic_gradient_descent(w,theta)
	# print(theta)
	return theta

def get_prediction(w,x):
	theta = get_parameters(w,len(x))
	prediction = numpy.dot(numpy.array(theta),numpy.array(x))
	return prediction

data_train = get_data('data_train.csv')
data_test = get_data('data_test.csv')

X_s,Y_s = arrange_data(data_train)
Xts,Yts = arrange_data(data_test)

# (HYPER-)PARAMETERS
tau = 0.1 # Weight Parameter
alpha = 0.01 # Learning Rate
max_n = 1000 # Stochastic Gradient Descent Loops
epsilon = 0.00000001 # Stochastic Gradient Descent Tolerance [not using here, though]

variance = float(0)
for i in range(len(Xts)):
	x = Xts[i]
	y = Yts[i]
	w = get_weights(X_s,x)
	prediction = get_prediction(w,x)
	print("Actual: " + str(y) + " Predicted: " + str(prediction))
	variance = variance + (prediction-y)**2
variance = variance/len(Xts)
print("Variance: ",variance)

