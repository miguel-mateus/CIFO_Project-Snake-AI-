from random import random
import random as rand
from operator import  attrgetter
from copy import deepcopy
import numpy as np
from copy import deepcopy
import csv
import time
from game import *
import math
from project_charles.FlatArray import flatener
from collections import Counter


class Individual:
    def __init__(
        self,
        representation=None,
        window_size=7, #Size of window around the snake's head
        hidden_size=15, #Size of hidden layers of the NN
        board_size=10, #Board size
        display=False #Wheter to print the games being played on the console
    ):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.board_size = board_size
        self.display = display

        if representation == None:
            brain = self.generate_brain(self.window_size**2, self.hidden_size, len(MOVES))
            self.representation = brain
        else:
            self.representation = representation

        self.fitness, self.score = self.evaluate()
        

    def generate_brain(self, input_size, hidden_size, output_size): #Initializes the random weigths of the individual's NN

        hidden_layer1 = np.array([[rand.uniform(-1, 1) for _ in range(input_size + 1)] for _ in range(hidden_size)])
        hidden_layer2 = np.array([[rand.uniform(-1, 1) for _ in range(hidden_size + 1)] for _ in range(hidden_size)])
        output_layer = np.array([[rand.uniform(-1, 1) for _ in range(hidden_size + 1)] for _ in range(output_size)])
        brain = [hidden_layer1, hidden_layer2, output_layer]
        return brain

    def evaluate(self): #Makes the individual play a game and returns both the fitness and score

        game = Game(self.board_size, 1, [self])
        outcome = game.play(False)

        #Prints a message if the individual died due to making 100 moves without eating an apple
        if outcome == 0: 
            print('Snake made it to the last turn')

        #Uncomment to use the second fitness function
        #moves = game.all_turns 

        score = len(game.snakes[0]) - 3

        #First fitness function
        fit = score + 1 

        #Second fitness function
        #fit = moves + ((2**score) + (score**2.1) * 500) - ((score**1.2) * (0.25*moves)**1.3) 
        
        return fit, score

    def get_move(self, board, snake): #Makes a forward pass in the NN and outputs a move

        input_vector = self.proccess_board(board, snake[-1][0], snake[-1][1])
        hidden_layer1 = self.representation[0]
        hidden_layer2 = self.representation[1]
        output_layer = self.representation[2]

        #Forward propagation of the neural network
        hidden_result1 = np.array([math.tanh(np.dot(input_vector, hidden_layer1[i])) for i in range(hidden_layer1.shape[0])] + [1]) # [1] for bias
        hidden_result2 = np.array([math.tanh(np.dot(hidden_result1, hidden_layer2[i])) for i in range(hidden_layer2.shape[0])] + [1]) # [1] for bias
        output_result = np.array([np.dot(hidden_result2, output_layer[i]) for i in range(output_layer.shape[0])])

        max_index = np.argmax(output_result)
        return MOVES[max_index]

    def proccess_board(self, board, x1, y1): #Creates the input to be used in the NN
        #x1 and y1 are the positions of the snake's head

        input_vector = [[0 for _ in range(self.window_size)] for _ in range(self.window_size)]

        for i in range(self.window_size):
            for j in range(self.window_size):
                ii = x1 + i - self.window_size//2
                jj = y1 + j - self.window_size//2

                #if the square is out of bounds
                if ii < 0 or jj < 0 or ii >= self.board_size or jj >= self.board_size:
                    input_vector[i][j] = -1

                #if there is food
                elif board[ii][jj] == FOOD:
                    input_vector[i][j] = 1

                #if there is nothing there
                elif board[ii][jj] == EMPTY:
                    input_vector[i][j] = 0

                #otherwise, there is part of the snake in that square
                else:
                    input_vector[i][j] = -1

        if self.display:
            print(np.array(input_vector))

        input_vector = list(np.array(input_vector).flatten()) + [1]
        return np.array(input_vector)

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, num_elite=10, window_size=7, hidden_size=15, board_size = 10, fit_sharing=False, mut_alpha=None, display_at_end=True, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        self.gen = 1
        self.timestamp = int(time.time())

        self.num_elite = num_elite #Number of elites
        self.fit_sharing = fit_sharing #Whether to use fitness sharing
        self.mut_alpha = mut_alpha #Alpha to be used for mutations that require it
        self.window_size = window_size #Size of window around snake head
        self.hidden_size = hidden_size #Size of hidden layers of the NN
        self.board_size = board_size #Size of the board
        self.ph_en = None #phenotypic entropy
        self.ph_va = None #phenotypic variance
        self.gt_en = None #genotypic entropy
        self.gt_va = None #genotypic variance
        self.display_at_end = display_at_end #Wheter to display the last generation playing the game

        for _ in range(size):
            self.individuals.append(
                Individual(window_size=self.window_size, hidden_size=self.hidden_size, board_size = self.board_size)
            )

    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism, log_title=''):

        if self.fit_sharing: #Performs fitness sharing in the first generation
            self.fitness_sharing()
        
        for gen in range(gens):
            new_pop = []

            #Saves elite individuals into the new generation
            if elitism == True:
                if self.optim=='max':
                    elite = deepcopy(self.individuals)
                    elite.sort(reverse = True, key=attrgetter('fitness'))
                    for i in range(self.num_elite):
                        new_pop.append(elite[i])

                if self.optim=='min':
                    elite = deepcopy(self.individuals)
                    elite.sort(reverse = False, key=attrgetter('fitness'))
                    for i in range(self.num_elite):
                        new_pop.append(elite[i])

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)
                while parent1 == parent2: #To prevent that both parents are the same individual
                    parent2 = select(self)
                # Crossover
                if  random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                # Mutation
                if random() < mu_p:
                    try:
                        offspring1 = mutate(offspring1, self.mut_alpha) 
                    except: #If the mutation doesn't require an alpha
                        offspring1 = mutate(offspring1)
                if random() < mu_p:
                    try:
                        offspring2 = mutate(offspring2, self.mut_alpha)
                    except: 
                        offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1, window_size=self.window_size, hidden_size=self.hidden_size, board_size = self.board_size))

                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2, window_size=self.window_size, hidden_size=self.hidden_size, board_size = self.board_size))
            
            #Removes the worst individuals until the population has the correct size
            if elitism == True:
                if self.optim=='max':
                    while len(new_pop) > self.size:
                        least = min(new_pop, key=attrgetter('fitness'))
                        new_pop.pop(new_pop.index(least))

                if self.optim=='min':
                    while len(new_pop) > self.size:
                        least = max(new_pop, key=attrgetter('fitness'))
                        new_pop.pop(new_pop.index(least))
            
            
            self.individuals = new_pop
            
            if self.optim=='max':

                scores = [] #Will hold all scores
                fit = [] #Will hold all fitnesses

                best_indiv = deepcopy(self.individuals)
                best_indiv.sort(reverse = True, key=attrgetter("fitness"))

                for i in range(self.size):
                    scores.append(self.individuals[i].score)
                    fit.append(self.individuals[i].fitness)

                scores.sort(reverse=True)

                self.ph_en = self.ph_entropy(fit)
                self.ph_va = self.ph_variance(fit)
                self.gt_en = self.gt_entropy()
                self.gt_va = self.gt_variance()

                self.log(log_title)

                print(scores)
                print('Ph Entropy:', self.ph_en, 'Ph Variance:', self.ph_va, 'Gt Entropy:', self.gt_en, 'Gt Variance:', self.gt_va)
                print('Gen', self.gen, f'Best Individual fitness: {best_indiv[0].fitness} apples: {best_indiv[0].score}')

            if self.optim=='min':

                scores = [] #Will hold all scores
                fit = [] #Will hold all fitnesses

                best_indiv = deepcopy(self.individuals)
                best_indiv.sort(reverse = False, key=attrgetter("fitness"))

                for i in range(self.size):
                    scores.append(self.individuals[i].score)
                    fit.append(self.individuals[i].fitness)

                scores.sort(reverse=True)

                self.ph_en = self.ph_entropy(fit)
                self.ph_va = self.ph_variance(fit)
                self.gt_en = self.gt_entropy()
                self.gt_va = self.gt_variance()

                self.log(log_title)

                print(scores)
                print('Ph Entropy:', self.ph_en, 'Ph Variance:', self.ph_va, 'Gt Entropy:', self.gt_en, 'Gt Variance:', self.gt_va)
                print('Gen', self.gen, f'Best Individual fitness: {best_indiv[0].fitness} apples: {best_indiv[0].score}')

            
            if self.fit_sharing:
                self.fitness_sharing()

            self.gen += 1

        self.log(log_title, all=False) #Place all=True to save the representation of the final population

        if self.display_at_end: #Display the algorithm playing snake at the end of the evolution

            #Saves all the individuals
            display_snakes = deepcopy(self.individuals)

            #Sorts them by fitness
            if self.optim=='max':
                display_snakes.sort(reverse = True, key=attrgetter('fitness'))
            
            if self.optim=='min':
                display_snakes.sort(reverse = False, key=attrgetter('fitness'))

            #Waits for any key to pressed on the console
            key = input('enter any character to display boards')

            #Starts displaying games, starting from the best individual
            for brain in display_snakes:
                self.display = True
                game = Game(brain.board_size, 1, [brain], display = True)
                gui = Gui(game, 800)
                game.play(True)
                print('Snake length', len(game.snakes[0]))


    def log(self, log_title, all=False):

        title = str(log_title + f'_run_{self.timestamp}.csv') #Title given to log file

        with open(title, 'a', newline='') as file:
            writer = csv.writer(file)
            if all:
                for i in self:
                    writer.writerow([self.gen, 'na', 'na', 'na', 'na', i.representation, i.fitness, i.score])
            else:
                elit = deepcopy(self.individuals)
                elit.sort(reverse = True, key=attrgetter('fitness'))
                writer.writerow([self.gen, self.ph_en, self.ph_va, self.gt_en, self.gt_va,'na', elit[0].fitness, elit[0].score])

    def fitness_sharing(self): #Performs fitness sharing

        dist = [] #Will hold "pseudo" distance matrix (without main diagonal)
        
        #Calculates all distances using euclidean distance
        for i in range(len(self.individuals)):
            dist_i = []
            for j in range(len(self.individuals)):
                if i != j: #No need to calculate the distance between one indiv and itself
                    euc = np.linalg.norm(flatener(self.individuals[i].representation)-flatener(self.individuals[j].representation)) #Distance between two indivs
                    dist_i.append(euc)
            dist.append(np.asarray(dist_i))

        dist = np.asarray(dist)

        mx = np.max(dist) #Max distance
        mn = np.min(dist) #Min distance

        dist = 1 - ((dist-mn) / (mx-mn)) #Sharing function (S(k)=1-k) applied to normalized distances

        for i in range(len(self.individuals)): #Iterates over all indivs
            sharing_coef = np.sum(dist[i]) #Calculates sharing coefficient
            self.individuals[i].fitness = self.individuals[i].fitness / sharing_coef #Redefine the fitness

    def ph_entropy(self, scores): #phenotypic entropy

        h = 0
        fitness_counter = Counter(scores).items() #Value counts of the different fitnesses

        for i in fitness_counter:
            f = i[1]/len(scores) #Fraction of individuals in the population having a certain fitness value
            h += f*np.log2(f)

        return h #phenotypic entropy of the population

    def ph_variance(self, scores): #phenotypic variance

        v = 0
        avg = np.mean(np.asarray(scores)) #Average fitness of the individuals in the population

        for i in scores:
            v += (i - avg)**2

        v = v/(len(scores) - 1)

        return v #phenotypic variance of the population

    def gt_entropy(self): #genotypic entropy (second technique)

        dist = [] #Will hold the distances from all individuals to the origin (defined here as the first individual in the pop, if elitism is used, it will be the best individual)
        
        #Calculates all distances using euclidean distance
        for i in range(1, len(self)):
            dist.append(np.linalg.norm(flatener(self[i].representation)-flatener(self[0].representation))) #Distance between indiv i and indiv origin
                        
        h = 0
        genotype_counter = Counter(dist).items() #Value counts of the distances

        for i in genotype_counter:
            f = i[1]/len(dist) #Fraction of individuals in the population having a certain distance
            h += f*np.log2(f)

        return h #genotypic entropy of the population

    def gt_variance(self): #genotypic variance

        dist = [] #Will hold the distances from all individuals to the origin (defined here as the first individual in the pop, if elitism is used, it will be the best individual)
        
        #Calculates all distances using euclidean distance
        for i in range(1, len(self)):
            dist.append(np.linalg.norm(flatener(self[i].representation)-flatener(self[0].representation))) #Distance between indiv i and indiv origin
        v = 0
        avg = np.mean(np.asarray(dist)) #Average distance of the individuals o the origin

        for i in dist:
            v += (i - avg)**2

        v = v/(len(dist) - 1)

        return v #phenotypic variance of the population  
        
    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"

    
