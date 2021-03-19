import statsmodels.api as sm 
from main import Drop7
import util, math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Any
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from scipy.optimize import curve_fit

def func(x,a,b):
    return a*np.log(x)+ b

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions: Callable, discount: float, featureExtractor: Callable, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.weight_plot = defaultdict(list)

    # Return the Q function associated with the weights and features
    def getQ(self, state: Tuple, action: Any) -> float:
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state: Tuple) -> Any:
        self.numIters += 1
        if self.explorationProb != 0:
            if random.random() < 1/(self.numIters**0.25):
                return random.choice(self.actions)
            else:
                return max((self.getQ(state, action), action) for action in self.actions)[1]
        else:
            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 1.0 / (self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple) -> None:
        w = self.weights
        phi = self.featureExtractor(state, action)
        eta = self.getStepSize()
        Q = self.getQ(state, action)
        v_opt = -1
        if newState:
            for cur_action in self.actions:
                v = self.getQ(newState, cur_action)
                v_opt = max(v, v_opt)
        else:
            v_opt = float(0)

        for feature in phi:
            key, val = feature
            self.weights[key] = w[key] - eta*((Q-(reward + self.discount*v_opt))*val +0.1*w[key]   )
            self.weight_plot[key].append(self.weights[key])
        



def get_new_group_size(state: Drop7, x: int, y: int) -> Tuple[int, int]:
    num_of_dets = 0
    det_score = 0
    if y < 7:
        group_size = 1
        if x != 0:
            group_size += state.groups_of_elements[x-1][y][0]
            index = group_size - 1
        else:
            index = 0
        if x != 6:
            group_size += state.groups_of_elements[x+1][y][0]
        for i in range(group_size):
            if state.field.elements[x - index + i][y] == group_size:
                num_of_dets +=   next_to_disc(state,x - index + i,y) + 1
                det_score   += y + 1
    else:
        return 0, 0, 0
    return group_size, num_of_dets, det_score,

def next_to_disc(state: Drop7, x: int, y: int) -> int:
    disc = 0
    if y < 7:
        if x > 0 and state.field.elements[x - 1][y] >=  8:
            disc += 1
            if state.field.elements[x - 1][y] == 8:
                disc += 1
        if y > 0 and state.field.elements[x][y - 1] >=  8:
            disc += 1
            if state.field.elements[x][y - 1] == 8:
                disc += 1
        if y < 6 and state.field.elements[x][y + 1] >=  8:
            disc += 1
            if state.field.elements[x][y + 1] == 8:
                disc += 1
        if x < 6 and state.field.elements[x + 1][y] >=  8:
            disc += 1
            if state.field.elements[x + 1][y] == 8:
                disc += 1
    else:
        return 0
    return disc

def next_to_disc_type(state: Drop7, x: int, y: int, disc_type: int) -> int:    
    if x > 0 and state.field.elements[x - 1][y] ==  disc_type:
        return True
    if x < 6 and state.field.elements[x + 1][y]  ==  disc_type:
        return True
    return False

def next_to_max_col(state: Drop7, x: int, max_val: int) -> int:
    if x > 0 and state.field.free_loc[x - 1] ==  max_val:
        return 1
    if x < 6 and state.field.free_loc[x + 1] ==  max_val:
        return 1
    return 0

def Drop7FeatureExtractor(state: Drop7, action: str) -> List[tuple]:
    y = state.field.free_loc[action]
    max_val = -1
    min_val = 8
    features = []
    col_dets = 0
    for col in range(7):
        if state.field.free_loc[col] > max_val:
            max_val = state.field.free_loc[col]
            max_col = col
        if state.field.free_loc[col] < min_val:
            min_val = state.field.free_loc[col]
            min_col = col

    num_of_max_col = state.field.free_loc.count(max_val) 
    group_size, row_dets, det_score = get_new_group_size(state, action, y)
    elem_det = (group_size == state.curr_elem) or ((y+1) ==  state.curr_elem)
    det_near_disc = elem_det and (det_score > 0)
    num_of_dets = row_dets
  
    if y < 7:
        for row in range(y):
            if state.field.elements[action][row]  == (y + 1) :
                col_dets  += next_to_disc(state,action,row) + 1
        col_dets += next_to_disc(state,action,y)
    if min_val == y:
        features.append((('min_eq_elem_{}'.format(min_val == y)), 1)) # True: action to drop disc on the min col.
    features.append((('row_dets'),          row_dets))  
    features.append((('col_dets'),          col_dets))
    if (max_val == y) and (state.field.free_loc.count(max_val) <= 2):
        features.append((('max_eq_elem'), col_dets)) # True: action to drop disc on the max col.
    if  (state.field.curr_elem == 1 and elem_det):
            features.append((('1_dets'), 1 + next_to_disc(state,action,y))) 
    if  elem_det:
            features.append((('elem_det'), 1 + next_to_disc(state,action,y))) 
        

        # if (max_val == y) and (col_dets == 0):
        #     features.append((('max_eq_val_no_dets', 1)))
    # features.append((('max_eq_elem_with_det_{}'.format((max_val == y) and (col_dets > 0))),1)) # True: action to drop disc on the max col.
    # features.append((('elem_det_{}'.format(             elem_det)),1)) # will the disc detonate
    # features.append((('det_near_disc_{}'.format(      det_near_disc)),1)) # will the disc detonate near breakable disks (blank disks)

    # features.append((('num_of_dets'),       num_of_dets/5))
    # features.append((('det_score',det_score),        1))
    # features.append((('y',                      y),1))

    # features.append((('next_to_max_col',        next_to_max_col(state, action, max_val) and det_near_disc),1))
    # features.append((('sum_of_elements'),   sum(state.field.free_loc)/55))
    # features.append((('y',                      y),1))
    # features.append((('min_col{}'.format (             min_col)),1)) #what col is the min col - may indacte how much the upper row is jagged  
    # features.append((('max_col{}'.format (              max_col)),1)) #what col is the max col - may indacte how much the upper row is jagged 
    ##################  Scrapped Features  ###################
    ##########################################################

    # features.append((('sum_of_elements', sum_of_elements),1))
    # features.append((('ocupied_row',tuple(max_lst), action),1))
    # features.append((('det_count', state.detonate_count),1))
    # features.append((('col_dets',               col_dets),1)) 
    # features.append((('next_to_max_col', ((max_col - 1) == action) or ((max_col + 1) == action)),1))
    # features.append((('detonate_by_col', ((state.field.free_loc[action]+1) ==  state.curr_elem)),1))
    # features.append((('end_game', (state.field.free_loc[action] ==  8)),1))
    # features.append((('end_game', (state.field.free_loc[action] ==  7)),1))
    # features.append((('end_game_lst', tuple(end_game_lst), action),1))
    # features.append((('end_game_flag', end_game_flag, action),1))
    # features.append((('next_to_min_col', ((min_col - 1) == action) or ((min_col + 1) == action)),1))

    return features




##########################################################
########################  Main  ##########################
##########################################################



#################################
############ Q Run ##############
#################################
print('start')
Q_learn = QLearningAlgorithm([0,1,2,3,4,5,6], 1, Drop7FeatureExtractor, explorationProb = 0.2)
Q_train_res = util.simulate(Q_learn, numTrials=500)
print(statistics.mean(Q_train_res))
print(statistics.stdev(Q_train_res))
df_weights = pd.DataFrame()
for key in Q_learn.weight_plot:
    df_weights = df_weights.append(pd.DataFrame({"Iteration":range(1,len(Q_learn.weight_plot[key])+1), "Weight": Q_learn.weight_plot[key], "Type":[key]*len(Q_learn.weight_plot[key])}))
# df_weights.to_pickle("tmp.pkl")
Q_learn.explorationProb = 0
Q_test_res = util.simulate(Q_learn, numTrials=100)
print(statistics.mean(Q_test_res))
print(statistics.stdev(Q_test_res))
util.simulate(Q_learn, numTrials=1, screen_en = True)
#################################
########## Save to DF ###########
#################################

df_train = pd.DataFrame({"Iteration":range(1,len(Q_train_res)+1), "Score": Q_train_res})
df_test = pd.DataFrame({"Iteration":range(1,len(Q_test_res)+1), "Score": Q_test_res})

# df_train.to_pickle("50kfinal_train_no_reg.pkl")
# df_test.to_pickle("50kfinal_test_no_reg.pkl")
# # df_train = pd.read_pickle('1400k02_eps_train.pkl')
# # df_test = pd.read_pickle('1400k02epsilon_train.pkl')

# print (y.mean())
# print (y.std())

#################################
############ LOG FIT ############
#################################
x = df_train["Iteration"]
y = df_train["Score"]
popt, pcov = curve_fit(func, x, y)
print(popt)

#################################
############ PLOT ###############
#################################
# sns.set_theme()
# fig1, axs1 = plt.subplots(ncols=1)
# sns.lineplot(data=df_weights, hue="Type",x="Iteration",y='Weight', legend="full")
# # sns.relplot(data=df_weights, x="Iteration", y="Weight", 
# #             col="Type", col_wrap=4,
# #             height=3, aspect=.75, linewidth=2.5,
# #             kind="line")
# # fig3, axs3 = plt.subplots(ncols=1)
# # sns.regplot(x=x, y=y, scatter = True, logx=True,  y_jitter=2, scatter_kws={"s": 2, 'alpha':0.15}, line_kws={'color':'g'})
# plt.show()
 