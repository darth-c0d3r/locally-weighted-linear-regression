import numpy

num_train = 100
num_test = 20

with open("data_train.csv",'a') as file:
	x = -5
	for i in range(num_train):
		y = (x)**2 + 0*numpy.random.normal(0,1)
		file.write(str(x)+","+str(y)+"\n")
		x = x + 10/num_train

with open("data_test.csv",'a') as file:
	x = -5
	for i in range(num_test):
		y = (x)**2 + 0*numpy.random.normal(0,1)
		file.write(str(x)+","+str(y)+"\n")
		x = x + 10/num_test