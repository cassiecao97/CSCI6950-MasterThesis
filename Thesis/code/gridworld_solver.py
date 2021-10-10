import gym
import numpy as np
import random
import pdb
from collections import defaultdict

class GridworldSolver(object):
    def __init__(self, num_feature=0):
        self.env = None
        self.env_name = ''
        self._init_environment(num_feature)
        self._policy = {}
        self._action_sequence = []
        self.state_ranges = self.env._get_state_ranges()

    def _init_environment(self, num_feature):
        assert num_feature not in [0,1] # We only consider more than 1 features
        self.env_name = 'gridworld-v%d' % (num_feature)

        self.env = gym.make(self.env_name)

        self.env._reset()

    # def compute_policy(self, start_state=None):

    #     # if start_state:
    #     #     self.env._set_start_state(start_state)

    #     max_action = 1

    #     while not self.env.finished:
    #     # for i in range(8):
    #         _, r, _, _ = self.env.step(max_action)
    #         print(f'Action Selected {max_action}\n')
    #         print(f'Reward for this step {r}\n')

    #         value_for_each_action = np.zeros(self.env.num_actions)
    #         prob_action_row = np.zeros(self.env.num_actions)
    #         s = self.env.agent_state

    #         print(s)
    #         for a in range(self.env.num_actions):
    #             prob_and_next_state = self.env.T(s, a)
                
    #             print("********************")
    #             print(f'action: {a}')
    #             for prob, s_prime in prob_and_next_state:
    #                 print((prob, s_prime.x, s_prime.y))
    #                 reward = self.env.R(s, a, s_prime)
    #                 print(f'reward: {reward}\n')
    #                 value_for_each_action[a] += prob * reward

    #         max_action = np.argmax(value_for_each_action)
    #         prob_action_row[max_action] = 1.0
    #         self._policy[s.__str__()] = prob_action_row
    #         self._action_sequence.append(max_action)

    def Q_learning(self, start_state=None):
        '''
        Compute the optimal policy given the default reward function R

        Inputs:
            None

        Outputs:
            None
        '''
        Q = defaultdict(lambda: np.zeros(self.env.num_actions))

        episode = 1000
        epsilon = 0.99 
        alpha = 0.95  

        for i in range(episode):
            # if i % 100 == 0: print("Starting episode %d" % i)

            state = self.env._reset()
        
            if epsilon > 0.1: epsilon *= 0.95
            if alpha > 0.05: alpha *= 0.95

            while not self.env.finished:
                if random.uniform(0,1) < epsilon:
                    action = random.choice(self.env.actions)
                else:
                    action = np.argmax(Q[state.__str__()])

                next_state, reward, finished, info = self.env.step(action)
    
                old_Q_value = Q[state.__str__()][action]
                next_max = np.max(Q[next_state.__str__()])

                new_Q_value = (1. - alpha) * old_Q_value + alpha * (reward + 0.99 * next_max - old_Q_value)
                
                Q[state.__str__()][action] = new_Q_value

                state = next_state
            
        self._policy = Q
        
        # pdb.set_trace()


    def policy_rollout(self, start_state=None, max_steps=float('inf')):
        ''' 
        Reset the environment, so current agent_state is set to agent base (where we always start at), then applies (roll out) the policy 

        Inputs:
            start_state     Where we start, default is None, indicating we start at agent_base
            max_steps       Number of steps into the future we would like to roll out our policy, default is infinity, meaning we go until we finish the entire environment

        Outputs:
            cumulative_reward       cumulative reward after we roll out our policy
            num_steps               number of timestamps we stepped
            exp_svf                 expected state visitation frequencies dictionary
        '''

        curr_state = self.env._reset()
        finished = self.env.finished

        # if start_state:
        #     self.env._set_start_state(start_state)

        num_steps = 0
        cumulative_reward = 0

        exp_svf = dict()
        exp_svf[curr_state.__str__()] = 1

        while not finished and num_steps < max_steps:

            next_state, reward, finished, info = self.env.step(self._get_action(curr_state.__str__()))

            # print(f'Current State: {curr_state}, Action: {self._get_action(curr_state.__str__())}, Next State: {next_state}, Reward: {reward}')

            cumulative_reward += reward

            curr_state = next_state

            num_steps += 1

            if curr_state.__str__() in exp_svf.keys():
                exp_svf[curr_state.__str__()] += 1
            else:
                exp_svf[curr_state.__str__()] = 1
        
        # print('\n')
        
        return cumulative_reward, num_steps, exp_svf

    def _set_policy(self, policy):
        ''' 
        Set the current policy to an input

        Input:
            policy      The policy we want to set our current policy to
        
        Ouput:
            None
        '''
        self._policy = policy
    
    def _get_policy(self):
        '''
        Return the current policy

        Input:
            None

        Output:
            None
        '''
        return self._policy
        
    def _get_action(self, state):
        '''
        Returns an action from the current policy at a given state.
        
        Input:
            state       state in a string formate, not an object format
        
        Output:
            action      the action to take at the given state, given the policy
        '''
        prob_dist = self._policy[state]
        action = np.argmax(prob_dist)

        return action
    
    def _get_actions(self):
        return self.env.actions
    
    def _get_transition_function(self):
        return self.env.T

    def _get_start_state(self):
        return self.env.agent_base

    def _get_state_feat_dict(self):
        return self.env.state_feat_dict
    
    def _get_feat_arr(self):
        return self.env.feat_arr

    def _get_action_sequence(self):
        return self._action_sequence