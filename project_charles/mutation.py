from random import uniform
import numpy as np
from project_charles.FlatArray import flatener,reshaper

#random mutation
def random_mut(individual, mut_percent = 0.1): 
  
    mutated=flatener(individual)
    for i in range(len(mutated)):
        if uniform(0,1)<mut_percent:
            val = uniform(-1,1)
            mutated[i]=val
    
    return reshaper(mutated)

#geometric semantic mutation
def geometric_mut(individual, mut_step): 
  
    mutated=flatener(individual)
    for i in range(len(mutated)):
        val = uniform(-mut_step,mut_step)
        mutated[i]=mutated[i]+val
    
    return reshaper(mutated)

#gaussian mutation
def gaussian_mut(individual, std): 
  
    mutated=flatener(individual)
    for i in range(len(mutated)):
        val = np.random.normal(0,std)
        mutated[i]=mutated[i]+val
    
    return reshaper(mutated)