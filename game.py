import numpy as np
import tkinter as tk
import time as time

#Global Variables
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

MOVES = [UP, DOWN, LEFT, RIGHT]

EMPTY = 0
FOOD = 99

class Game:

    def __init__(self, size, num_snakes, players, gui=None, display=False, max_turns=100):
        self.size = size #Size of the game board
        self.num_snakes = num_snakes #Number of snakes on the board, (The implementation we adapted the snake game from allowed for more than one snake on the board)
        self.players = players #Individual object controling the snake
        self.gui = gui #Whether to show the game being played by an individual in a window or not
        self.display = display #Whether to print chosen move and the game board in every turn
        self.max_turns = max_turns #Maximum number of turns an individual can take to eat the next apple

        self.num_food = 4 #Number of initial apples
        self.turn = 0 #Will count the number of turns (moves) an individual takes without eating an apple
        self.all_turns = 0 #Will count the total number of turns (moves) an individual takes playing the game
        self.snake_size = 3 #Initial size of the snake
        
        self.snakes = [[((j + 1)*self.size//(2*self.num_snakes), self.size//2 + i) for i in range(self.snake_size)]
                        for j in range(self.num_snakes)]

        #Initial positions of the apples
        self.food = [(self.size//4, self.size//4), (3*self.size//4, self.size//4),
                    (self.size//4, 3*self.size//4), (3*self.size//4, 3*self.size//4)]

        self.player_ids = [i for i in range(self.num_snakes)]

        self.board = np.zeros([self.size, self.size]) #Initialization of the game board
        for i in self.player_ids:
            for up in self.snakes[i]:
                self.board[up[0]][up[1]] = i + 1
        for tup in self.food:
            self.board[tup[0]][tup[1]] = FOOD

        self.food_index = 0
        #Positions of the apples for the entire game
        self.food_xy = [(6, 1), (4, 4), (2, 9), (0, 6), (8, 6), (8, 9), (1, 1), (7, 5), (9, 7), (3, 7), (8, 9), (1, 3), (7, 9), (2, 3), (9, 9), (8, 3), (2, 2), (7, 9), (5, 3), (0, 2), (8, 6), (7, 7), (1, 7), (9, 1), (6, 7), (8, 7), (8, 3), (7, 3), (5, 8), (5, 6), (2, 4), (7, 5), (1, 3), (5, 4), (8, 4), (1, 3), (0, 4), (8, 5), (6, 6), (7, 9), (8, 9), (4, 7), (4, 4), (9, 3), (2, 5), (1, 4), (8, 5), (1, 3), (7, 1), (2, 1), (8, 4), (1, 2), (3, 1), (9, 6), (8, 6), (6, 0), (6, 3), (7, 6), (7, 6), (5, 0), (5, 0), (3, 4), (1, 3), (2, 8), (0, 9), (7, 0), (0, 5), (3, 5), (3, 3), (3, 4), (9, 3), (1, 9), (7, 3), (3, 8), (2, 6), (2, 3), (0, 8), (9, 5), (8, 1), (6, 7), (8, 6), (1, 6), (2, 6), (3, 9), (7, 6), 
(6, 8), (2, 4), (2, 0), (3, 2), (1, 0), (9, 7), (3, 6), (3, 7), (5, 8), (0, 8), (3, 3), (2, 8), (2, 8), (1, 4), (6, 1), (1, 5), (8, 3), (7, 9), (0, 2), (6, 1), (1, 9), (7, 3), (8, 5), (7, 3), (7, 1), (0, 0), (7, 6), (1, 9), (6, 4), (8, 8), (4, 6), (4, 8), (4, 1), (6, 8), (8, 3), (8, 3), (5, 3), (9, 6), (9, 6), (8, 0), (3, 8), (3, 0), (1, 5), (3, 5), (1, 0), (2, 1), (2, 1), (1, 5), (0, 6), (5, 7), (2, 1), (0, 6), (1, 3), (3, 0), (5, 5), (0, 3), (0, 3), (4, 5), (2, 6), (7, 7), (1, 8), (6, 1), (0, 1), (9, 6), (0, 9), (6, 8), (1, 0), (7, 0), (1, 7), (2, 7), (1, 4), (8, 2), (3, 3), (1, 4), (3, 3), (3, 5), (1, 0), (7, 0), (5, 6), (8, 0), (7, 5), (7, 6), (2, 2), (5, 9), (6, 2), (3, 3), (1, 9), (5, 3), (9, 6), (2, 2), (4, 7), (2, 7), (3, 9), (1, 7), (4, 9), (4, 3), (9, 6), (5, 8), (7, 5), (8, 4), (8, 3), (2, 1), (9, 7), (7, 0), (6, 0), (2, 7), (2, 1), (3, 8), (6, 2), (4, 7), (7, 0), (8, 7), (0, 0), (1, 
7), (0, 2)]

    def move(self):
        moves = []
        #Move the head
        for i in self.player_ids:
           snake_i = self.snakes[i]
           move_i = self.players[i].get_move(self.board, snake_i)
           moves.append(move_i) 
           new_square = (snake_i[-1][0] + move_i[0], snake_i[-1][1] + move_i[1])
           snake_i.append(new_square)
        #Update the tail
        for i in self.player_ids:
            head_i = self.snakes[i][-1]
            if head_i not in self.food:
                self.board[self.snakes[i][0][0]][self.snakes[i][0][1]] = EMPTY
                self.snakes[i].pop(0)
            else:
                self.food.remove(head_i)
                self.turn=0

        #Check for out of bounds
        for i in self.player_ids:
            head_i = self.snakes[i][-1]
            if head_i[0] >= self.size or head_i[1] >= self.size or head_i[0] < 0 or head_i[1] < 0:
                self.player_ids.remove(i)
            else:
                self.board[head_i[0]][head_i[1]] = i + 1
        
        #Check for collisions
        for i in self.player_ids:
            head_i = self.snakes[i][-1]
            for j in range(self.num_snakes):
                if i == j:
                    if head_i in self.snakes[i][:-1]:
                        self.player_ids.remove(i)
                else:
                    if head_i in self.snakes[j]:
                        self.player_ids.remove(i)

        #Spawn new food
        while len(self.food) < self.num_food:
            x = self.food_xy[self.food_index][0]
            y = self.food_xy[self.food_index][1]
            while self.board[x][y] != EMPTY:
                self.food_index += 1
                x = self.food_xy[self.food_index][0]
                y = self.food_xy[self.food_index][1]
            self.food.append((x, y))
            self.board[x][y] = FOOD
            self.food_index += 1
        return moves

    def play(self, display=False):
        if display:
            self.display_board()
        while True:
            if len(self.player_ids) == 0:
                return -1
            if self.turn >= self.max_turns:
                return 0
            moves = self.move()
            self.turn += 1
            self.all_turns += 1
            if display:
                for move in moves:
                    if move == UP:
                        print('UP')
                    elif move == RIGHT:
                        print('RIGHT')
                    elif move == LEFT:
                        print('LEFT')
                    else:
                        print('DOWN')

                self.display_board()
                if self.gui is not None:
                    self.gui.update()
                time.sleep(0.5)

    def display_board(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == EMPTY:
                    print('|_', end='')
                elif self.board[i][j] == FOOD:
                    print('|#', end='')
                else:
                    print('|' + str(int(self.board[i][j])), end='')
            print('|')

class Gui: #Class used to create an object that displays a game being played in a new window

    def __init__(self, game, size):
        self.game = game
        self.game.gui = self
        self.size = size
        
        self.ratio = self.size / self.game.size
        
        self.app = tk.Tk()
        self.canvas = tk.Canvas(self.app, width=self.size, heigh=self.size)
        self.canvas.pack()

        for i in range(len(self.game.snakes)):
            color = '#' + '{0:03X}'.format((i + 1)*500)
            snake = self.game.snakes[i]
            self.canvas.create_rectangle(self.ratio*(snake[-1][1]), self.ratio*(snake[-1][0]),
                                         self.ratio*(snake[-1][1] + 1), self.ratio*(snake[-1][0] + 1), fill = color)

            for j in range (len(snake) - 1):
                color = '#' + '{0:03X}'.format((i + 1)*123)
                snake = self.game.snakes[i]
                self.canvas.create_rectangle(self.ratio*(snake[j][1]), self.ratio*(snake[j][0]),
                                            self.ratio*(snake[j][1] + 1), self.ratio*(snake[j][0] + 1), fill = color)

        for food in self.game.food:
            self.canvas.create_rectangle(self.ratio*(food[1]), self.ratio*(food[0]),
                                         self.ratio*(food[1] + 1), self.ratio*(food[0] + 1), fill = '#000000000')

    def update(self):
        self.canvas.delete('all')
        for i in range(len(self.game.snakes)):
            color = '#' + '{0:03X}'.format((i + 1)*500)
            snake = self.game.snakes[i]
            self.canvas.create_rectangle(self.ratio*(snake[-1][1]), self.ratio*(snake[-1][0]),
                                         self.ratio*(snake[-1][1] + 1), self.ratio*(snake[-1][0] + 1), fill = color)

            for j in range (len(snake) - 1):
                color = '#' + '{0:03X}'.format((i + 1)*123)
                snake = self.game.snakes[i]
                self.canvas.create_rectangle(self.ratio*(snake[j][1]), self.ratio*(snake[j][0]),
                                            self.ratio*(snake[j][1] + 1), self.ratio*(snake[j][0] + 1), fill = color)

        for food in self.game.food:
            self.canvas.create_rectangle(self.ratio*(food[1]), self.ratio*(food[0]),
                                         self.ratio*(food[1] + 1), self.ratio*(food[0] + 1), fill = '#000000000')
        self.canvas.pack()
        self.app.update()
