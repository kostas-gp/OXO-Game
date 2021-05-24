
  
# Game class

import numpy as np
import random
class TicTacToe:
    player_names = {0: ' ', 1: 'x', -1: 'o'} # printed player names
    def __init__(self): 
        """initialization
        """
        self.turn = 0 # current turn
        self.players = [1, -1] # here 1, -1 represent the x, o players
        self.board = np.zeros((3,3)) # game board
        
    def reset(self): 
        """resets the board
        """
        self.turn = 0 # current turn
        self.board = np.zeros((3,3)) # game board
        
    def get_turn(self):
        return self.players[(self.turn % 2)]
    
    def play(self, position): 
        """ plays a turn at the specified position
        """
        x, y = position[0], position[1]
        player = self.get_turn()
        self.board[x,y] = player
        self.turn += 1
        end, win = self.check_end(), self.check_winner()
        if win != 0:
            end = True
        return (end, win)
    
    def available_actions(self):
        """returns all the open positions
        """
        empty = np.where(self.board == 0)
        return list(zip (empty[0],empty[1]))
    
    def get_state(self):
        return "".join(
        [game.player_names[x] for x in game.board.ravel()]
                      )
    
    def check_end(self):
        return (np.sum(abs(game.board)) == 9)
    
    def check_winner(self):
        """check whether there is a winner
        """
        diag_sum = np.trace(self.board)
        second_diag_sum = np.trace(np.fliplr(self.board))
        hor_sum = np.sum(self.board, axis=0)
        ver_sum = np.sum(self.board, axis=1)
        for p in self.players:
            p_sum = 3*p
            if (
                any(hor_sum == p_sum) or
                any(ver_sum == p_sum) or
                diag_sum == p_sum or
                second_diag_sum == p_sum
            ):
                return p
            return 0
        
        
    def _is_available(self, x, y):
        """checks if x,y position is available
        for the player's turn
        """
        return (self.board[x,y] == 0)
    
    def __str__(self):
        """ print board
        """
        out = ""
        for i in range(len(self.board)):
            out += "-------------\n"
            out += "| " + " | ".join(
                [str(game.player_names[x]) for x in game.board[i,:]]
                ) + " |\n"
            out += "-------------\n"
            return out
			
# Random Player initialize a player

class RandomPlayer:
    def __init__(self, game, id):
        self.id = id
        self.game = game
    
    def act(self):
        available_action = self.game.available_actions()
        return available_action[np.random.choice(range(len(available_action)))]
		
#Simulation 1 in order to print the board

game = TicTacToe()
for i in range(5):
    game.reset()
    while True:
        action = random.choice(game.available_actions())
        end, win = game.play(action)
        if end or win != 0:
            print(game)
            break
            print(end, win)
			
#Simulation 2 # 100 randon games

game = TicTacToe()
player_1 = RandomPlayer(game, 1)
player_2 = RandomPlayer(game, -1)
games = 100
score = 0.0
for i in range(games):
    game.reset()
    end = False
    while not end:
        turn = game.get_turn()
        if turn == 1: # player 1 plays
            action = player_1.act()
        else: # player 2 plays
            action = player_2.act()
            pass
        end, win = game.play(action)
        if end:
            score += win
            break
print(score / games)

#Class Q-Learning algorithm

class QLearner:
    """class that implements q-learnin"""
    
    def __init__(self, epsilon=0.1, learning_rate=0.1, discount_factor=0.7):
        self.Q = dict() # Q table
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
    def e_greedy(self, actions):
        """ epsilon greedy action selection
        """
        if np.random.random() < self.epsilon:
            return random.choice(actions.keys())
        else:
            max_action = None
            max_value = -float('Inf')
            for action, value in actions.items():
                if value > max_value:
                    max_action, max_value = action, value
                    return max_action
    
    def insert_state(self, state, possible_actions):
        if state not in self.Q: # state has not been visited so far
            self.Q[state] = dict() # initialize
            for action in possible_actions:
                self.Q[state][action] = 0.0
                
    def act(self, state, possible_actions):
        """acts for input state
        """
        self.insert_state(state, possible_actions)
        # available actions in this state
        actions = self.Q[state]
        # return egreedy action
        return self.e_greedy(actions)
    
    def update(self, state, action, next_state, next_state_actions, reward,terminal=False):
        """update rule for q learning"""
        self.insert_state(next_state, next_state_actions)
        best_next_value = 0.0
        if self.Q[next_state].values():
            best_next_value = max(self.Q[next_state].values())
            self.Q[state][action] += self.learning_rate * (
                reward + self.discount_factor * best_next_value
                ) - self.Q[state][action]
            
#class agent 

class LearnerPlayer:
    def __init__(self, game, player_id):
        self.id = player_id
        self.game = game
        self.brain = QLearner()
        self.previous_state = None
        self.previous_action = None
        
    def reset(self):
        self.previous_state = None
        self.previous_action = None
        
    def act(self):
        state = self.game.get_state()
        available_actions = self.game.available_actions()
        action = self.brain.act(state, available_actions)
        self.previous_state = state
        self.previous_action = action
        return action
    
    def learn(self, reward):
        current_state = self.game.get_state() # get current state
        current_state_actions = self.game.available_actions()
        if self.previous_state: # if there is previous state update
            self.brain.update(
                self.previous_state,
                self.previous_action,
                current_state,
                current_state_actions,
                reward
            )
			
#Simulation of a random player with the ahent

game = TicTacToe()
player_1 = RandomPlayer(game, 1)
player_2 = LearnerPlayer(game, -1)
games = 100000 # how many games to play
score = 0.0
score_history = []
for i in range(games):
    game.reset() # reset game
    player_2.reset() # reset learner
    end = False
    while not end: # play one game until is over
        turn = game.get_turn()
        if turn == 1: # player 1 plays
            action = player_1.act()
        else: # player 2 plays
            action = player_2.act()
            pass
        end, win = game.play(action) # apply action to the game, here the state changes
        if end or turn == 1: # that' a hack for the learning step of player 2
            player_2.learn(-win)
            if end:
                score += win
                score_history.append(score / float(i+1))
                break
print(score / games)

# Graph

from matplotlib import pyplot as plt
plt.plot(range(games), score_history)
plt.ylabel('Avg. score')
plt.xlabel('Episodes')
plt.show()

