from random import uniform
from operator import attrgetter
from random import sample

#Fitness Proportionate Selection
def fps(population):

    total_fitness = sum([i.fitness for i in population])
    
    spin = uniform(0, total_fitness)
    position = 0
    
    for individual in population:
        position += individual.fitness
        if position > spin:
            return individual

#Tournament Selection
def tournament(population, size = 20):

    tournament = sample(population.individuals, size)

    if population.optim == 'max':
        return max(tournament, key=attrgetter('fitness'))
    elif population.optim == 'min':
        return max(tournament, key=attrgetter('fitness'))
    else:
        raise Exception('No optimization specified (min or max),')

#Rank Selection
def rank(population):
    
    if population.optim == 'max':
        population.individuals.sort(key=attrgetter('fitness'))
    elif population.optim == 'min':
        population.individuals.sort(key=attrgetter('fitness'), reverse=True)

    
    total = sum(range(population.size+1))
    
    spin = uniform(0, total)
    position = 0
    
    for count, individual in enumerate(population):
        position += count + 1
        if position > spin:
            return individual