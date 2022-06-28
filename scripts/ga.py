import numpy as np
import pickle
from numpy.random import randint
from numpy.random import rand
import random
import matplotlib.pyplot as plt
# load dictionary with tfidf values
file = open("data/tfidf.pkl","rb")
tfidf_dict = pickle.load(file)
def decorate_axis(ax, remove_left=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.grid(axis='x', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

# return number of non zero elements
def nnz(x):
	return np.count_nonzero(x)

def tfidf_avg(x):
	ids = np.where(np.array(x)==1)[0]
	avg = 0
	for i in range (ids.shape[0]):

		avg+= tfidf_dict[str(ids[i])]
	return avg/i

def obj_fun(x):
	selected = nnz(x)

	if selected < 1000:
		return 0
	else:
		tf = tfidf_avg(x)
		return tf -((selected - 1000)/8520 * tf)


# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] > scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# double point crossover
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover points that is not on the end of the string
		pt_1 = randint(1, len(p1)-2)
		pt_2 = randint(pt_1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt_1] + p2[pt_1:pt_2] + p1[pt_2:]
		c2 = p2[:pt_1] + p1[pt_1:pt_2] + p2[pt_2:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, id):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]

    # keep track of best solution
	best, best_eval = 0, objective(pop[0])
	best_vec = np.array([])
    # enumerate generations
	best_cnt = 0
	for gen in range(n_iter):

		flag = False
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		prev = best_eval
        # check for new best solution
		for i in range(n_pop):
			if scores[i] > best_eval:
				best_cnt = 0

				best, best_eval = pop[i], scores[i]
				# print("new best %f" % ( scores[i]))

		best_vec = np.append(best_vec, best_eval)

        # select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]

        # create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
		if(0< (best_eval - prev)/prev * 100 < 0.001 ):
			break
		# break conditions
		# best is the same for 50 generations
		if(best_cnt == 50):
			break

		best_cnt = best_cnt + 1

	gens_vec = np.arange(0, gen+1)
	fig = plt.figure()
	ax0 = fig.add_subplot()
	decorate_axis(ax0)
	ax0.plot(gens_vec, best_vec, c="slateblue")
	ax0.set_title("Pop Size: " + str(n_pop) + " Cross R: "+ str(r_cross)+" Mut R: " + str(r_mut))
	ax0.set_xlabel("Generation")
	ax0.set_ylabel("Fitness")
	plt.savefig(str(id)+".png")
	return [best, best_eval, gen]

# define the total iterations
n_iter = 1000

# bits
n_bits = 8520

# define the population size
n_pop = [20, 20, 20, 20, 20, 200, 200, 200, 200, 200]
# crossover rate
r_cross = [0.6, 0.6, 0.6, 0.9, 0.1, 0.6, 0.6, 0.6, 0.9, 0.1]
# mutation rate
r_mut = [0.00, 0.01, 0.10, 0.01, 0.01, 0.00, 0.01, 0.10, 0.01, 0.01]

f = open('data/results.dat', 'w')


gen_avg = np.zeros_like(n_pop)
score_avg = np.zeros_like(n_pop)
for i in range(0,5):
	for it in range (0, len(n_pop)):
		# perform the genetic algorithm search
		print(it)
		best, score, gen = genetic_algorithm(obj_fun, n_bits, n_iter, n_pop[it], r_cross[it], r_mut[it], it)
		with open(str(it)+'.npy', 'wb') as f:
			np.save(f, best)
		print(score, gen)
		print('Done!')
		gen_avg[it]+= gen/5
		score_avg[it]+= score/5
print(gen_avg)
print(score_avg)