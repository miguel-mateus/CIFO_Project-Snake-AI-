from random import uniform
from project_charles.FlatArray import flatener,reshaper

#Arithemetic Crossover
def arithmetic_co(p1, p2):

    p1=flatener(p1)
    p2=flatener(p2)

    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    alpha = uniform(0,1)
    
    for i in range(len(p1)):
        offspring1[i] = p1[i] * alpha + (1-alpha) * p2[i]

        offspring2[i] = p1[i] * (1-alpha) + alpha * p2[i]

    return reshaper(offspring1), reshaper(offspring2)

#Geometric Crossover
def geometric_co(p1, p2):

    p1=flatener(p1)
    p2=flatener(p2)

    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)
    
    for i in range(len(p1)):

        alpha = uniform(0,1)

        offspring1[i] = p1[i] * alpha + (1-alpha) * p2[i]

        offspring2[i] = p1[i] * (1-alpha) + alpha * p2[i]

    return reshaper(offspring1), reshaper(offspring2)

#Simulated binary crossover
def sbc(p1, p2, n=2): 

    p1=flatener(p1)
    p2=flatener(p2)

    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)
    
    u = uniform(0,1)

    if u <= 0.5:
        b = (2*u)**(1/(n+1))
    else:
        b = (1/(2*(1-u)))**(1/(n+1))

    offspring1 = 0.5*((1+b)*p1 + (1-b)*p2)
    offspring2 = 0.5*((1+b)*p2 + (1-b)*p1)

    return reshaper(offspring1), reshaper(offspring2)