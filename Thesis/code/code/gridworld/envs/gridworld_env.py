import gym
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding


class CellState:
    def __init__(self, x, y, c, feat_count=[]):
        self.x = x
        self.y = y
        self.feature = c
        self.feat_count = feat_count
    
    def __str__(self):
        return f'coordinate:({self.x} {self.y}), feature:{self.feature}, feature_counts:{self.feat_count}'

# ---------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------#

class GridworldEnv(gym.Env):
    def __init__(self, map_file, transition_noise=0):
        self.finished = False

        self.rewards = {'R': 2, 'B': 10}

        self.update_state_feat_dict = False

        # total_bins, action_state_dict, state_feat_dict, feat_arr, these four variables are updated in _read_grid_map()
        self.total_bins = None
        self.action_state_dict = {} # {2: CellState(), 3: CellState(), ...}, Pickup(0) and GoBase(1) will not be in it
        self.state_feat_dict = {} # {(x0,y0):feature, (x1,y1):feature, ...}, this gets updated when a pickup action happens
        self.feat_arr = []

        self.curr_path = os.path.dirname(os.path.realpath(__file__))
        self.map_path = os.path.join(self.curr_path, map_file)
        self.start_grid_map = self._read_grid_map(self.map_path) # 2D Array of CellState objects
        self.start_grid_map = self._config_grid_map(self.start_grid_map)

        self.actions = self._get_actions() # 0: Pickup 1: GoS1(S1 is base) 2: GoS2 3: GoS3 4: GoS4
        self.action_space = spaces.Discrete(len(self.actions))

        # print(self.action_state_dict)
        # print(self.feat_arr)
        # print(self.actions)
        # print(self.start_grid_map)

        self.agent_base = self._get_agent_base(self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_base)

        self.grid_map_shape = self.start_grid_map.shape # (2, 3)
        self.num_actions = len(self.actions)
        self._transition_noise = transition_noise
    
    def _get_actions(self):
        a = [i for i in range(len(self.action_state_dict) + 2)]

        return a

    def _get_state_ranges(self):
        return np.array([[0, self.grid_map_shape[0]], [0, self.grid_map_shape[1]]]) # [[0,2], [0,3]]

    def step(self, action, nn=None):
        ''' 
        return next state, reward, finish status, info 
        '''

        if self.finished: return self.agent_state, 0, True, {"success":True}

        info = {}
        curr_state = copy.deepcopy(self.agent_state)

        # Sample from T(s,a) for next states
        if self.T(curr_state, action):
            prob, next_state = self.T(curr_state, action)[0] # sampled_next_states = [(prob, state), (prob, state), ...]
        else:
            info["success"] = False
            raise Exception("No valid next state in transition function")
        
        # update agent_state to next_state
        self.agent_state = copy.deepcopy(next_state)

        if not nn:
            # We are using our default reward function R to find the reward of curr_state
            reward = self.R(curr_state, action, next_state)

            if self.update_state_feat_dict:
                self.state_feat_dict[(curr_state.x, curr_state.y)] = None # remove the bin
                self.update_state_feat_dict = False
        else:
            # We are using our neural network to find the reward of curr_state
            reward = nn[0].forward(nn[1])

            # print(reward.requires_grad)

            # if the current action is a pickup action
            if action == 0 and (curr_state.x, curr_state.y) != (self.agent_base.x, self.agent_base.y):
                # then we need to check whether we are updating the curr_state such that it shows already picked up
                if self._check_if_valid_pick_up(curr_state):
                    self.state_feat_dict[(curr_state.x, curr_state.y)] = None # remove the bin
        
        # check if we are done exploring the envrionment 
        self.finished = self.is_terminal()
        info["success"] = True

        return self.agent_state, reward, self.finished, info

    def R(self, state, action, next_state):

        if state == next_state:
            return -1

        if action == 0:
            if self._check_if_valid_pick_up(state):
                self.update_state_feat_dict = True
                if state.feature == "R": return self.rewards['R']
                else: return self.rewards['B']
            else:
                return -1
        elif action == 1:
            return 1
        else:
            if self._check_if_valid_pick_up(next_state):
                if next_state.feature == 'R':
                    return (self.rewards['R']/sum(self.rewards.values()))
                elif next_state.feature == 'B':
                    return (self.rewards['B']/sum(self.rewards.values()))
            else: return -1

        # if action == 0: # pickup
        #     if (state.x, state.y) == (self.agent_base.x, self.agent_base.y): # at base
        #         return -1
        #     else: # at a bin
        #         if self._check_if_valid_pick_up(state):
        #             self.update_state_feat_dict = True
        #             if state.feature == "R": return self.rewards['R']
        #             else: return self.rewards['B']
        #         else:
        #             return -1
        # elif action == 1: # GoS1(Go to base)
        #     if (state.x, state.y) == (self.agent_base.x, self.agent_base.y): # at base
        #         return -1
        #     else: # at a bin
        #         return 1
        # else: # GoS2, GoS3, GoS4
        #     if (state.x, state.y) != (self.agent_base.x, self.agent_base.y): # at a bin
        #         return -1
        #     else: # at base
        #         if self._check_if_valid_pick_up(next_state):
        #             if next_state.feature == 'R':
        #                 return (self.rewards['R']/sum(self.rewards.values()))
        #             elif next_state.feature == 'B':
        #                 return (self.rewards['B']/sum(self.rewards.values()))
        #         else: return -1

    def T(self, state, action, next_state=None):
        ''' 
        self-transition for invalid action
        returns an array of (prob, next state) tuple 
        '''

        if not next_state:
            if action == 0: # pickup
                if not state.feature: # if the current state is agent base or if it has already been picked up, then we transit from current_state -> current_state
                    return [(1.0 - self._transition_noise, state)]
                else: # if we are at a state that is still active to be PICKEUP
                    return [(1.0 - self._transition_noise, self._transit_state(state))]
            else: # GoS1, GoS2, GoS3, GoS4
                probability_of_next_states = self.next_states_and_prob(state, action) # returns all(prob, next states) that the robot can go to, next states与action相关

                return probability_of_next_states
        else:
            probability_of_next_states = self.next_states_and_prob(state, action)

            for outcome in probability_of_next_states:
                if outcome[1] == next_state:
                    return [(outcome[0], next_state)]

            return [(0, next_state)]

    def next_states_and_prob(self, curr_state, curr_action):
        agent_base_coord = (self.agent_base.x, self.agent_base.y)
        curr_state_coord = (curr_state.x, curr_state.y)

        # At base, performing a GO action
        if curr_state_coord == agent_base_coord:
            if curr_action == 1: # GOS1(Go to base), invalid
                return [(1.0 - self._transition_noise, curr_state)]

            prob_and_states = [] # (prob, state) tuples

            x, y = self.action_state_dict[curr_action].x, self.action_state_dict[curr_action].y
            feature_in_that_state_now = self.state_feat_dict[(x, y)]
            target_next_state = CellState(x, y, feature_in_that_state_now, curr_state.feat_count)

            prob_and_states = [(1.0 - self._transition_noise, target_next_state)]

            if self._transition_noise > 0:
                if curr_action == 2: # if we are at base and select GoS2
                    other_action = [3, 4]
                elif curr_action == 3: # if we are at base and select GoS3
                    other_action = [2, 4]
                else: # if we are at base and select GoS4
                    other_action = [2, 3]

                for a in other_action:
                    pos_next_state = CellState(self.action_state_dict[a].x, self.action_state_dict[a].y, self.action_state_dict[a].feature, curr_state.feat_count)
                    prob_and_states.append((self._transition_noise/len(other_action), pos_next_state))
            
            return prob_and_states
        # At a bin, performing a GO action
        else:
            if curr_action == 1:
                next_state = CellState(self.agent_base.x, self.agent_base.y, None, curr_state.feat_count)
                return [(1.0 - self._transition_noise, next_state)]
            else:
                return [(1.0 - self._transition_noise, curr_state)]

    def _transit_state(self, curr_state):
        new_state = copy.deepcopy(curr_state)

        for i, f in enumerate(self.feat_arr):
            if curr_state.feature == f:
                new_state.feat_count[i] += 1

        new_state.feature = None

        return new_state

    def _check_if_valid_pick_up(self, curr_state):
        if self.state_feat_dict[(curr_state.x, curr_state.y)]: # check if still valid to be picked up
            return True
        
        return False

    def is_terminal(self):
        ''' check if at terminal state (all bins are picked up) '''

        curr_state = self._get_agent_state()

        # print(curr_state.num_red)
        # print(curr_state.num_blue)

        if sum(curr_state.feat_count) == self.total_bins:
            return True

        return False

    def _read_grid_map(self, map_path):
        ''' read txt grid map into 2D grid map, also count how many bins are there '''
        with open(map_path, 'r') as map_file:
            grid_map = map_file.readlines()

        num_row = len(grid_map)
        num_col = len(grid_map[0].split())

        grid = [[None for x in range(num_col)] for y in range(num_row)]
        total_bins = 0
        action_id = 2
        
        for r_num, line in enumerate(grid_map):
            # values = [int(x) for x in line.split()]
            values = line.split()
            row = []
            for i, v in enumerate(values):
                if v == 'X':
                    row.append(CellState(r_num, i, None))
                elif v != '0':
                    cell = CellState(r_num, i, v)
                    row.append(cell)

                    self.action_state_dict[action_id] = cell
                    self.state_feat_dict[(r_num, i)] = v

                    total_bins += 1
                    action_id += 1

                    if v not in self.feat_arr:
                        self.feat_arr.append(v)
                else:
                    row.append(None)
            
            grid[r_num][:] = row

        self.total_bins = total_bins
        grid = np.array(grid)

        return grid
    
    def _config_grid_map(self, map):
        for i in range(len(map)):
            for j in range(len(map[i])):
                if map[i][j]:
                    map[i][j].feat_count = np.zeros(len(self.feat_arr), int)

        return map

    def _get_agent_base(self, map):
        ''' get current map's start state/base '''

        for i in range(len(map)):
            for j in range(len(map[i])):
                cell = map[i][j]
                if cell and not cell.feature:
                    return cell

        raise Exception("Agent Base State not specified")

        return None

    def _get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state
    
    # def _set_start_state(self, state):
    #     pass

    def _reset(self):
        self.finished = False
        self.total_bins = None

        self.start_grid_map = self._read_grid_map(self.map_path)
        self.start_grid_map = self._config_grid_map(self.start_grid_map)

        self.agent_base = self._get_agent_base(self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_base)

        return self.agent_state

# def _2Dmap_to_observation(self, grid_map):
#     pass

# def render(self):
#     pass

# def _close_env(self):
#     plt.close(1)
#     return
