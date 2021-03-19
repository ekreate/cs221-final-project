from scipy.optimize import curve_fit
import scipy.stats as stats
import curses
import statistics
import time
import sys
from random import randint
import random
import collections
from copy import deepcopy
from tabulate import tabulate
from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    
class Field:
    def __init__(self, size):
        self.size = size
        self.icons = {
            0: '  .  ',
            1: '  1  ',
            2: '  2  ',
            3: '  3  ',
            4: '  4  ',
            5: '  5  ',
            6: '  6  ',
            7: '  7  ',
            8: '  © ',
            9: '  o  ',
        }
        self.elements =  [[ 0 for i in range(7) ] for j in range(7)]
        self.free_loc =   [ 0 for i in range(7) ]
        self.curr_free_loc =   [ 0 for i in range(7) ]  
        self.curr_elem = 0
        self.score= 0

    def render(self, screen):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED,       curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN,     curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW,    curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_BLUE,      curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_CYAN,      curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_MAGENTA,   curses.COLOR_BLACK)
        # curses.init_pair(7, curses.COLOR_PAIRS,     curses.COLOR_BLACK)
        screen.addstr(1, 28, "DROP7")
        screen.addstr(2, 21,  'Score:{0} ---- Drop:{1}'.format(self.score, self.curr_elem) )
        for i in range(7):
            row = ''
            for j in range(7):
                row = self.icons[self.elements[j][i] ]
                if self.elements[j][i] < 7:
                    screen.addstr(10-i, j*4+16 , row, curses.color_pair(self.elements[j][i]))
                else:
                    screen.addstr(10-i, j*4+16 , row)

class Drop7:
    def __init__(self):
        self.groups_of_elements =  [[ [0,0] for i in range(7) ] for j in range(7)]
        self.curr_groups_of_elements = None
        self.curr_elem = random.randint(1,7)
        self.x = None
        self.y = None
        self.level_up_flag = False
        self.split_group = False
        self.update_needed = None
        self.del_queue = deque()
        self.endGame = False
        self.iteration = 0
        self.detonate_count = 0

    def get_elem_drop_loc(self, ch):
        #human mode
        if ch >= ord('1') and ch <= ord('7'):
            self.drop_loc = (int(chr(ch)) - 1)
        #AI mode
        else:
            self.drop_loc = ch

    def level_up(self):
        self.level_up_flag = True
        for i in range(7):
            self.field.free_loc[i] += 1
            self.field.curr_free_loc[i] += 1
            if self.field.free_loc[i] == 8:
                self.endGame = True
                return self.field.score
            self.field.elements[i].pop()
            self.field.elements[i].insert(0, 9)
            self.curr_groups_of_elements[i].pop()
            self.curr_groups_of_elements[i].insert(0,[7,i])
        self.level_up_flag = False
        self.update_entire_board()

    def del_element(self, x, y):
        self.field.curr_free_loc[x] -= 1
        self.del_queue.append((x,y))
       
    def drop_elem(self):
        self.x = self.drop_loc
        self.y = self.field.free_loc[self.drop_loc]
        self.detonate_count = 0
        if self.y == 7:
            self.endGame = True
            return self.field.score
        self.field.elements[self.x][self.y] =  self.curr_elem 
        self.field.free_loc[self.drop_loc] += 1
        self.field.curr_free_loc[self.drop_loc] += 1
        self.field.score += 1
        self.update_row_groups(self.x, self.y)
        self.curr_groups_of_elements = deepcopy(self.groups_of_elements)
        self.update_entire_board()
        self.curr_elem = random.randint(1,7)
        self.field.curr_elem = self.curr_elem

    def del_all_elements(self):
        update_cols = deque()
        while(len(self.del_queue) != 0):
            x,y = self.del_queue.pop()
            update_cols.append(x)
            self.field.elements[x][y] = 'x'
        while(len(update_cols) != 0):  
            x = update_cols.pop()
            for i in range(7):
                if self.field.elements[x][i] == 'x':
                    del self.field.elements[x][i]
                    self.field.elements[x].append(0)

    def clean_groups(self):
        for x in range(7):
            for y in range(self.field.curr_free_loc[x], self.field.free_loc[x], 1):
                    self.split_row_groups(x, y)
                        # with open('log.txt', 'a') as f:
                        #     print("{},{}\n".format(x,y), file=f)

    def update_entire_board(self):
        while(True):
            self.update_needed = False
            # with open('log.txt', 'a') as f:
            #     print("elements: \n", file=f)
            #     print(tabulate(self.field.elements, headers=['0','1','2','3','4','5','6']), file=f)
            #     print("\ngroups: \n", file=f)
            #     print(tabulate(self.curr_groups_of_elements, headers=['0','1','2','3','4','5','6']), file=f)
            for x in range(7):
                for y in range(self.field.free_loc[x]):
                    self.check_for_groups(x,y)
            for x in range(7):
                for y in range(self.field.free_loc[x]):
                    self.check_for_col(x,y)
            self.del_all_elements()
            self.clean_groups()
            self.groups_of_elements = deepcopy(self.curr_groups_of_elements)
            self.field.free_loc = deepcopy(self.field.curr_free_loc)
            if self.update_needed is False:
                break
              
    def update_row_groups(self, x, y):
        group_size = 1
        if x != 0:
            group_size += self.groups_of_elements[x-1][y][0]
            index = group_size - 1
            self.groups_of_elements[x][y][1] = index
        else:
            index = 0
        if x != 6:
            group_size += self.groups_of_elements[x+1][y][0]
        for i in range(group_size):
            self.groups_of_elements[x - index + i][y][0] = group_size
            self.groups_of_elements[x - index + i][y][1] = i

    def split_row_groups(self, x, y):
        left_group_size = self.curr_groups_of_elements[x][y][1]
        right_group_size = self.curr_groups_of_elements[x][y][0] - (self.curr_groups_of_elements[x][y][1] + 1)
        self.curr_groups_of_elements[x][y] = [0,0]
        for i in range(left_group_size):
            curr_idx = x - left_group_size + i
            self.curr_groups_of_elements[curr_idx][y][0] = left_group_size
            self.curr_groups_of_elements[curr_idx][y][1] = i
        for i in range(right_group_size):
            curr_idx = x + i + 1
            self.curr_groups_of_elements[curr_idx][y][0] = right_group_size
            self.curr_groups_of_elements[curr_idx][y][1] = i
     
    def check_for_groups(self, x, y):
        if self.field.elements[x][y] == self.field.free_loc[x]:
            if (x,y) not in self.del_queue:
                self.detonate(x,y) 

    def check_for_col(self, x, y):
        group_size = self.groups_of_elements[x][y][0]
        if (self.field.elements[x][y] == group_size) and (group_size != 0):
            if (x,y) not in self.del_queue:
                self.detonate(x,y) 

    def detonate(self, x, y):
        # self.detonate_count += 1
        self.update_needed = True
        self.del_element(x,y)
        if x > 0 and self.field.elements[x - 1][y] >=  8:
            if self.field.elements[x - 1][y] == 9:
                self.field.elements[x - 1][y] -= 1
            else:
                self.field.elements[x - 1][y] = random.randint(1,7)
        if y > 0 and self.field.elements[x][y - 1] >=  8:
            if self.field.elements[x][y - 1]  == 9:
                self.field.elements[x][y - 1]  -= 1
            else:
                self.field.elements[x][y - 1]  = random.randint(1,7)
        if x < 6 and self.field.elements[x + 1][y] >=  8:
            if self.field.elements[x + 1][y] == 9:
                self.field.elements[x + 1][y] -= 1
            else:
                self.field.elements[x + 1][y] = random.randint(1,7)
        if y < 6 and self.field.elements[x][y + 1] >=  8:
            if self.field.elements[x][y + 1] == 9:
               self.field.elements[x][y + 1] -= 1
            else:
                self.field.elements[x][y + 1] = random.randint(1,7)
        # with open('log.txt', 'a') as f:
        #         print("\n**detonate {},{}**: \n".format(x,y), file=f)
        #         print(tabulate(self.field.elements, headers=['0','1','2','3','4','5','6']), file=f)
    
    def set_field(self, field):
        self.field = field
        field.curr_elem = self.curr_elem


def human(screen):
    global score
    screen.timeout(0)
    field = Field(7)
    game = Drop7()
    game.set_field(field)
    i = 0
    while(True):
        field.render(screen)
        screen.refresh()
        # ch = random.randint(0,6)
        ch = screen.getch()
        if ch != -1:
            game.get_elem_drop_loc(ch)
            score = game.drop_elem()
            if score is not None:
                break
            i+=1
            if i%5 == 0:
                score = game.level_up()
                if score is not None:
                    break
        time.sleep(.2)

def ai():
    global score
    field = Field(7)
    game = Drop7()
    game.set_field(field)
    i = 0
    while(True):
        ch = random.randint(0,6) # ai here
        game.get_elem_drop_loc(ch)
        score = game.drop_elem()
        if score is not None:
            break
        i+=1
        if i%5 == 0:
            score = game.level_up()
            if score is not None:
                break


def playDrop7(screen, drop7, action):
    if screen != None:
        drop7.field.render(screen)
        screen.refresh()
        screen.getch()
    drop7.get_elem_drop_loc(action)
    drop7.drop_elem()
    drop7.iteration += 1
    if (drop7.iteration % 5) == 0:
        drop7.level_up()
    return drop7

if __name__=='__main__':
    # #human
    # curses.wrapper(human)
    # print("\n\n***the score is {}***".format(score))
    #ai:
    field = Field(7)
    state = Drop7()
    state.set_field(field)
    scores = []
    i =0 
    while i < 1000:
        playDrop7(None, state, random.choice([0,1,2,3,4,5,6]))
        if state.endGame == True:
            i += 1
            scores.append(state.field.score)
            field = Field(7)
            state = Drop7()
            state.set_field(field)
    print(statistics.mean(scores))
    print(statistics.stdev(scores))
    
    # df_rand = pd.DataFrame({"Iteration":range(1,len(scores)+1), "Score": scores, "Agent":["Random"]*1000}) # df random
    # df_tot = df_rand
    df_rl = pd.read_pickle('50kfinal_test.pkl')
    df_tot = pd.read_pickle('50kfinal_test_no_reg.pkl')
    df_tot["Agent"] = ["RL - No Regulazation"]*1000 
    df_rl["Agent"] = ["RL"]*1000 
    df_tot = df_tot.append(df_rl, ignore_index=True)
    print(statistics.mean(df_rl["Score"]))
    print(statistics.stdev(df_rl["Score"]))
    # df_a = pd.DataFrame({"Iteration":range(1,5001), "Score": [73.2]*5000}) # df hooman
    fig, axs = plt.subplots(ncols=1)
    fig, ax = plt.subplots()
    g = sns.histplot(data=df_tot, hue="Agent", x="Score", kde=True, legend=None)
    # sigma = 21.39
    # mu = 73.2
    # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    # y = 1000* stats.norm.pdf(x, mu, sigma)
    # sns.lineplot(x=x, y=y,color="green", legend="full")
    # g.legend_.remove()
    # plt.legend()
    # handles, labels = ax.get_legend_handles_labels()
    # patch = mpatches.Patch(color='green', label='Human')
    # handles.append(patch) 
    # plt.legend(handles=handles)
    # sns.histplot(data=df_tot, hue="Agent", x="Score", kde=True,)
    # sns.regplot(x=x, y=y, scatter = True, x_bins=20, x_ci='sd', scatter_kws={"s": 2, 'alpha':0.15}, line_kws={'color':'tan'})
    # # ax.errorbar(x, y, yerr, solid_capstyle='projecting', capsize=5)
    # # df_a = pd.read_pickle('1400k02_eps_train.pkl')
    # df_a = pd.read_pickle('50kfinal_train.pkl')
    # x = df_a["Iteration"]
    # y = df_a["Score"]
    # print (y.mean())
    # print (y.std())
    # # popt, pcov = curve_fit(func, x, y)
    # # print(popt)
    # # fig, axs = plt.subplots(ncols=1)
    # plt.show()
    # sns.regplot(x=x, y=y, scatter = True, logx=True,  y_jitter=2, scatter_kws={"s": 2, 'alpha':0.01}, line_kws={'color':'g'})
    # axes = plt.axes()
    # axes.set_ylim([20, 80])
    # # axs.set_xscale('log')
    plt.show()
    # df_a = pd.read_pickle('50k02_eps_test.pkl')
    # x = df_a["Iteration"]
    # y = df_a["Score"]
    # mean_1 = np.array([73.2]*5000)
    # std_1 = np.array([21.39]*5000)
    # mean_2 = np.array([statistics.mean(scores)]*5000)
    # std_2 = np.array([statistics.stdev(scores)]*5000)
    # rl = np.array(y)
    # rl_mean = np.array(np.array([statistics.mean(y)]*5000))
    # std_rl = statistics.stdev(y)
    # x = np.arange(len(mean_1))
    # sns.set()
    # plt.plot(x, mean_1, 'b-', label='Human')
    # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.1)
    # plt.plot(x, mean_2, 'r-', label='Random')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.1)
    # plt.plot(x, rl, 'k.', markevery= 9,  alpha=0.1)
    # plt.plot(x, rl_mean, 'g-', label='Q-learning', linewidth=2)
    # plt.fill_between(x, rl_mean - std_rl, rl_mean + std_rl, color='g', alpha=0.2)
    # plt.xlabel('Iteration')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.show()
