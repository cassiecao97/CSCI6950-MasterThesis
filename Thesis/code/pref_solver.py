import copy
import time
import os
import numpy as np
import argparse
from deep_maxent_irl import *
from gridworld_solver import *
from gridworld.envs.gridworld_env import CellState


PARSER = argparse.ArgumentParser(description="Deep Maxent IRL Parameters")
PARSER.add_argument('--n_feat', dest='n_feat', action="store", default=2, type=int, help="Number of features in your current environment")
PARSER.add_argument('--gamma', dest='gamma', action="store", default=0.99, type=float, help="Discount Factor") # gamma
PARSER.add_argument('--n_iters', dest='n_iters', action="store", default=200, type=int, help="Number of iterations to tune NN") # n_iters
PARSER.add_argument('--traj_len', dest='traj_len', action="store", default=6, type=int, help="The length of each demonstrated trajectory") # len_traj
PARSER.add_argument('--lr', dest='learning_rate', action="store", default=0.95, type=float, help="Learning Rate") # learning_rate
ARGS = PARSER.parse_args()

GAMMA = ARGS.gamma
N_ITERS = ARGS.n_iters
TIMESTAMP = ARGS.traj_len
N_FEAT = ARGS.n_feat
LR = ARGS.learning_rate

# ---------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------#

def get_expert_policy(path, start, f_s_dict, feat_arr, actions):
    policies = []
    trajs = []

    input_file = open(path, 'r')

    for line in input_file:
        traj = [start]
        pi = dict()

        cur_state = copy.deepcopy(start)
        
        line = line.split()
        for i in range(len(line)-1):
            if line[i+1] == 'P':
                action = actions[0]

                ind = feat_arr.index(cur_state.feature)

                new_feat_count = copy.deepcopy(cur_state.feat_count)
                new_feat_count[ind] = new_feat_count[ind] + 1
                next_state = CellState(cur_state.x, cur_state.y, None, new_feat_count)

            elif line[i+1] == 'X':
                action = actions[1]
                next_state = CellState(start.x, start.y, None, cur_state.feat_count)

            else:
                ind = feat_arr.index(line[i+1])
                action = actions[ind+2]

                # because we use f_s_dict, we must guarantee that one feature only appears once in the entire environment
                # this is one of the drawbacks of this code example
                next_state = CellState(f_s_dict[line[i+1]][0], f_s_dict[line[i+1]][1], line[i+1], cur_state.feat_count)

            pi[cur_state.__str__()] = action
            traj.append(next_state)

            cur_state = next_state

        if i == len(line)-2:
            # If we explore the whole envrionment, then we will always end with P, when feat_count = [1,1], then the corresponding action at this state should be GoBase 
            pi[cur_state.__str__()] = actions[1]

            next_state = CellState(start.x, start.y, None, cur_state.feat_count)
            traj.append(next_state)

        policies.append(pi)
        trajs.append(traj)
    
    return policies, trajs

# ---------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    ''' 
    Two features map 
    '''
    solver = GridworldSolver(N_FEAT)
    feat_state_dict = {v : k for k,v in solver._get_state_feat_dict().items()}
    # print(feat_state_dict)

    ''' 
    Compute optimal policy given R 
    '''
    start_time = time.time()
    # solver.compute_policy()
    solver.Q_learning()
    elapsed_time = time.time() - start_time
    # print("\nFinished Computing Policy in %g seconds\n" % elapsed_time)

    agent_policy = solver._get_policy()
    print(f'Optimal Policy Given R: {agent_policy}\n')

    cumulative_reward, num_steps, _ = solver.policy_rollout()
    print(f'Optimal Policy Cumulative Reward: {cumulative_reward}\n')

    # print("--------------------------------------------\n")

    ''' 
    Get expert demonstration 
    '''
    file_path = os.getcwd() + "/Thesis/code/expert_demos.txt"
    expert_policies, trajs = get_expert_policy(file_path, solver._get_start_state(), feat_state_dict, solver._get_feat_arr(), solver._get_actions())
    print(f'Expert Policies: {expert_policies}\n')
    print(f'Expert Trajectories: {trajs}\n')

    print("--------------------------------------------\n")

    ''' 
    Contruct the initial feat_map with states in demos
    '''
    feat_n_col = (len(solver._get_feat_arr())+3)*2+len(solver._get_actions())
    feat_map = np.empty((0, feat_n_col), int)
    
    #     None = 0, feat_arr = ['R', 'B'] then R = 1, B = 2
    #     Action is a one-hot vector
    #     Next state is a vector of 1x5
    #     feat_map is a matrix of dimension Nx15

    #     feat_map = [[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    #                 [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    #                 [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    #                 [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0]
    #                 [0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1]
    #                 [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    for i, traj in enumerate(trajs):
        for j in range(len(traj)-1):
            state = traj[j]

            if not state.feature:
                cur_state = np.concatenate((np.array([state.x, state.y, 0]), state.feat_count), axis=None)
            else:
                cur_state = np.concatenate((np.array([state.x, state.y, solver._get_feat_arr().index(state.feature)+1]), state.feat_count), axis=None)
            
            action_ind = expert_policies[i][state.__str__()]
            action_vector = np.zeros(shape=len(solver._get_actions()))
            action_vector[action_ind] = 1

            if j == 0:
                prev_state_vector = np.concatenate((cur_state,action_vector), axis=None)
            else:
                feat_map = np.vstack([feat_map, np.concatenate((prev_state_vector, cur_state), axis=None)])
                prev_state_vector = np.concatenate((cur_state, action_vector), axis=None)
        
        state = traj[-1]
        if not state.feature:
            cur_state = np.concatenate((np.array([state.x, state.y, 0]), state.feat_count), axis=None)
        else:
            cur_state = np.concatenate((np.array([state.x, state.y, solver._get_feat_arr().index(state.feature)+1]), state.feat_count), axis=None)

        feat_map = np.vstack([feat_map, np.concatenate((prev_state_vector, cur_state), axis=None)])

    '''
    Build Neural Network 
    '''
    fcnn = FCNN(feat_map.shape[1], 2, [20, 10])

    # print("--------------------------------------------\n")

    ''' 
    Deep Max Entropy IRL Training 
    '''
    T = solver._get_transition_function()
    feat_arr = solver._get_feat_arr()
    IRL = DeepMaxEntIRL(feat_map, T, trajs, fcnn, feat_arr)
    R_hat = IRL.deep_maxent_irl()

    # start_time = time.time()

    # elapsed_time = time.time() - start_time
    # print("Finished Policy Ranking in %g seconds" % elapsed_time)
