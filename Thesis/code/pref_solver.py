import copy
import time
import os
import numpy as np
import argparse
from deep_maxent_irl import *
from gridworld.envs.gridworld_env import CellState
from gridworld_solver import GridworldSolver


PARSER = argparse.ArgumentParser(description="Deep Maxent IRL Parameters")
PARSER.add_argument('--n_feat', dest='n_feat', action="store", default=2, type=int, help="Number of features in your current environment")
PARSER.add_argument('--gamma', dest='gamma', action="store", default=0.99, type=float, help="Discount Factor") # gamma
PARSER.add_argument('--n_iters', dest='n_iters', action="store", default=1, type=int, help="Number of iterations to tune NN") # n_iters
PARSER.add_argument('--traj_len', dest='traj_len', action="store", default=6, type=int, help="The length of each demonstrated trajectory") # len_traj
PARSER.add_argument('--lr', dest='learning_rate', action="store", default=0.05, type=float, help="Learning Rate") # learning_rate
ARGS = PARSER.parse_args()

GAMMA = ARGS.gamma
N_ITERS = ARGS.n_iters
TIMESTAMP = ARGS.traj_len
N_FEAT = ARGS.n_feat
LR = ARGS.learning_rate

def get_expert_demonstrations(path, start, f_s_dict, feat_arr):
    trajs_str = []
    trajs = []

    input_file = open(path, 'r')

    for line in input_file:
        traj_str = [start.__str__()]
        traj = [start]

        cur_state = copy.deepcopy(start)

        for s in line.split():
            if s == 'X':
                next_state = CellState(start.x, start.y, None, cur_state.feat_count)
            elif s == 'P':
                for i, f in enumerate(feat_arr):
                    if cur_state.feature == f:
                        break

                new_feat_count = copy.deepcopy(cur_state.feat_count)
                new_feat_count[i] = new_feat_count[i] + 1
                next_state = CellState(cur_state.x, cur_state.y, None, new_feat_count)
            else:
                # because we use f_s_dict, we must guarantee that one feature only appears once in the entire environment
                # this is one of the drawbacks of this code example
                next_state = CellState(f_s_dict[s][0], f_s_dict[s][1], s, cur_state.feat_count)

            traj_str.append(next_state.__str__())
            traj.append(next_state)

            cur_state = next_state

        trajs_str.append(traj_str)
        trajs.append(traj)
    
    return trajs_str, trajs

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
    print("\nFinished Computing Policy in %g seconds\n" % elapsed_time)

    # actions = solver._get_action_sequence()
    # print(actions)

    agent_policy = solver._get_policy()
    print(f'Optimal Policy Given R: {agent_policy}\n')

    cumulative_reward, num_steps, _ = solver.policy_rollout()
    print(f'Optimal Policy Cumulative Reward: {cumulative_reward}\n')

    ''' 
    Get expert demonstration 
    '''
    file_path = os.getcwd() + "/expert_demos.txt"
    expert_demos, trajs = get_expert_demonstrations(file_path, solver._get_start_state(), feat_state_dict, solver._get_feat_arr())
    print(f'Expert Demonstrations in readable strs: {expert_demos}\n')
    print(f'Expert Demonstrations in CellState Objects: {trajs}\n')

    ''' 
    Contruct the initial feat_map with states in demos
    '''
    feat_map = np.empty((0, len(solver._get_feat_arr())+3), int)
    
    # None = 0, feat_arr = ['R', 'B'] then R = 1, B = 2
    # feat_map = [[1, 1, 0, 0, 0]
    #             [0, 0, 1, 0, 0]
    #             [0, 0, 0, 1, 0]
    #             [1, 1, 0, 1, 0]
    #             [0, 1, 2, 1, 0]
    #             [0, 1, 0, 1, 1]]
    for traj in trajs:
        for state in traj:
            if not state.feature:
                state_vector = np.concatenate((np.array([state.x, state.y, 0]), state.feat_count), axis=None)
            else:
                state_vector = np.concatenate((np.array([state.x, state.y, solver._get_feat_arr().index(state.feature)+1]), state.feat_count), axis=None)
            
            
            feat_map = np.vstack([feat_map, state_vector])

    '''
    Build Neural Network 
    '''
    fcnn = FCNN(feat_map.shape[1], 4, [400, 300, 200, 200])

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
