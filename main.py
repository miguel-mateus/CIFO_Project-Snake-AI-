from project_charles.charles import *
from game import *
from project_charles.selection import *
from project_charles.mutation import *
from project_charles.crossover import *
from NN_Parameters import *

title = 'Log_values' #Name of the log files
num_runs = 5 #Number of runs to perform

for i in range(num_runs):

    pop = Population(
        size=100, #Size of the population
        optim='max', #Type of optimization
        num_elite=1, #Number of elites
        fit_sharing=True, #Whether or not to use fitness sharing
        mut_alpha = 0.25, #Value to be used either as mutation step in geometric mutation, std in gaussian mutation or mutation percentage in random percentage
                          #Place as false if another mutation function is used
        window_size = window_size, #Size of the window around the snakes head
        hidden_size = hidden_size, #Number of neurons in each hidden layer
                                   #Both can be changed on the file NN_Parameters.py
        board_size = 10, #Size of the game board
        display_at_end = False #Whether to display the last generation of individuals playing the Snake Game, if more than one run is being performed, placing this as True will interrupt the runs
    )

    log_title = title + '_V' + str(i + 1)

    pop.evolve(
        gens=500, #Number of generations
        select= tournament, #Selection method
        crossover= arithmetic_co, #Crossover operator
        mutate=geometric_mut, #Mutation operator
        co_p=0.9, #Crossover rate
        mu_p=0.2, #Mutation rate
        elitism=True, #Whether or not to use elitism
        log_title = log_title
    )