"""
Universidade Federal de Campina Grande - UFCG
PPGCC - Programa de Pós-graduação em Ciência da Computação
Class: Machine Learning
Student: Ruan Victor Bertoldo Reis de Amorim
"""
from numpy import *

def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y - (m * x + b)) **2
	return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
	#gradient descent
	gradient = compute_gradient_rss(b_current, m_current, points)
	new_b = b_current - (learningRate * gradient[0])
	new_m = m_current - (learningRate * gradient[1])
	return [new_b, new_m]

def compute_norm_gradient_rss(b_current, m_current, points):	
	grandient = compute_gradient_rss(b_current, m_current, points)
	return math.sqrt((grandient[0] **2) + (grandient[1] **2)) 	 

def compute_gradient_rss(b_current, m_current, points):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
		m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
	
	return (b_gradient, m_gradient)

def gradient_descent_runner(points, starting_b, starting_m, learning_rate):
	b = starting_b
	m = starting_m
	
	#tolerance criterion
	tolerance = 0.1
	
	while (compute_norm_gradient_rss(b, m, points ) >= tolerance):
		b,m = step_gradient(b, m, array(points), learning_rate)
		# printing RSS for each iteration
		print(compute_error_for_given_points(b, m, points))
		
	return [b, m]

def run():
	points = genfromtxt('income.csv', delimiter=',')
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	#num_iterations = 16000
	
	print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, points)))
	print ("Running...")
	[b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate)
	print ("Result: b = {0}, m = {1}, error = {2}".format(b, m, compute_error_for_given_points(b, m, points)))

if __name__ == '__main__':
	run()
