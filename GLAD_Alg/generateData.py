import numpy

def sample_accuracies(n):
	# random number with mean 1 and variance 1
	return numpy.random.randn(numLabelers) + 1

def sample_inv_difficulties(n):
	# random number according to exp(normal(1,1))
	return numpy.exp(numpy.random.randn(numItems) + 1)

if __name__ == "__main__":
	numLabelers = 10
	numItems = 20

	# generate random labelers and images
	alpha = sample_accuracies(numLabelers)
	beta = sample_inv_difficulties(numItems)
	
	# True labels
	# random label 0 or 1
	z = numpy.random.random_integers(0,1, numItems)

	# calculate the probabilities of each label being correct
	A, B = numpy.meshgrid(alpha,beta)
	_, Z = numpy.meshgrid(alpha,z)
	probabilities = 1/(1+numpy.exp(-A*B))

	# generate observed labels
	temp = numpy.random.rand(*probabilities.shape)
	labels = (temp < probabilities)*Z + (temp >= probabilities)*(1-Z)
	print labels