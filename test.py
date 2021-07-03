import random as rand

#Code used to produce the fixed positions of the apples for all of the snake games
x = [(rand.randint(0, 9), rand.randint(0, 9)) for _ in range(200)]

print(x)

