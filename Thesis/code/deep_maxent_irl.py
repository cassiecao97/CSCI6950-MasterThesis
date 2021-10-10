from typing import OrderedDict
import numpy as np
import torch
import random
from torch import nn
from torch._C import dtype
from torch.nn import *
from collections import OrderedDict, defaultdict, Counter
from pref_solver import GAMMA, LR, N_ITERS, TIMESTAMP, N_FEAT
from gridworld.envs.gridworld_env import CellState
from gridworld_solver import GridworldSolver


class FCNN(nn.Module):
    def __init__(self, n_input, n_layers, hiddenDim):
        super().__init__()

        self.n_input = n_input
        self.n_layers = n_layers
        self.hiddenDim = hiddenDim

        self.model, self.theta = self._build_network()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5)

        print(self.model)

    def _build_network(self):
        '''
        Create and return a Seqential Model

        Input
            feat_map        a NxD vector representing a state, with N equal to the number of states

        Output
            model           a Pytorch Sequential Model with n_layers number of hidden layers
            theta           a list of model parameters that will be trained by an optimizer
                            (the collection of variables or training parameters which should be modified when minimizing the loss)
        '''
        layers = OrderedDict()

        inp = self.n_input

        for i in range(self.n_layers):
            layers["fc"+str(i+1)] = nn.Linear(inp, self.hiddenDim[i])
            layers["relu"+str(i+1)] = nn.ReLU()

            inp = self.hiddenDim[i]

        layers["output"] = nn.Linear(inp, 1)

        model = nn.Sequential(layers)
        theta = list(model.parameters())
        
        return model, theta

    def forward(self, state_tensor):
        '''
        define forward pass

        Input
            state_tensor        a 1xD vector representing a state

        Output
            reward              reward of the state
        '''

        # input = torch.from_numpy(state_tensor).type(torch.FloatTensor)
        input = torch.tensor(state_tensor, dtype=torch.float)

        reward = self.model(input)

        return reward

    def backward(self, demo_svf, exp_svf):

        # a = torch.tensor(np.array([1, 1, 1, 1, 1, 1, 0, 0]), dtype=float)
        # b = torch.tensor(np.array([1, 1, 5, 0, 0, 0, 0, 0]))
        # c = torch.tensor(np.array([1, 0, 0, 0, 0, 0, 1, 5]))

        # print(f'Loss for (b, a): {self.loss_fn(b, a)}')
        # print(f'Loss for (c, a): {self.loss_fn(c, a)}')

        label = np.array([v for v in demo_svf.values()])
        states = np.array([v for v in demo_svf.keys()])
        for state in exp_svf.keys():
            if state not in demo_svf.keys():
                label = np.append(label, 0)
                states = np.append(states, state)
        label = torch.tensor(label, dtype=float, requires_grad=True)
        
        pred = np.array([])
        for state in states:
            if state in exp_svf.keys():
                pred = np.append(pred, exp_svf[state])
            else:
                pred = np.append(pred, 0)
        pred = torch.tensor(pred, requires_grad=True)

        loss = self.loss_fn(pred, label)

        # Zero the gradiants accumulated from the previous steps
        self.optimizer.zero_grad()

        # Perform backpropagation
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        print(f'{label}\n')
        print(f'{pred}\n')
        print(f'exp_svf {exp_svf}\n')
        print(f'Loss: {loss}\n')

        return loss

# ---------------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------------#


class DeepMaxEntIRL():
    def __init__(self, f, T, trajs, fcnn, feat_arr):
        self.feat_map = f
        self.T = T
        self.demo_trajs = trajs
        self.fcnn = fcnn
        self.feat_arr = feat_arr

        self.epsilon = 0.99
        self.lr = LR
        self.solver = GridworldSolver(N_FEAT)

    def demo_trajs_svf(self):
        '''
        Compute state visitation frequencies from expert demonstrations

        Inputs:
            None

        Outputs:
            demo_svf     State visitation frequencies of demonstrations
        '''
        demo_svf = {}
        for traj in self.demo_trajs:
            for state in traj:
                if state.__str__() in demo_svf.keys():
                    demo_svf[state.__str__()] += 1
                else:
                    demo_svf[state.__str__()] = 1

        # {CellStateObject1 : 1, CellStateObject2: 1, ...}
        demo_svf = {k : int(v/len(self.demo_trajs)) for k, v in demo_svf.items()}

        return demo_svf

    def get_input_vector(self, s, a, s_prime):
        '''
        Change the states [0, 0, None, 0, 0] to a tensor that can be passed through neural network

        Input;
            s       a state
            a       action
            s_prime     next state

        Output:
            state_vector    a tensor that could be used as an input to the neural network
        '''

        if not s.feature:
            s_vector = np.concatenate((np.array([s.x, s.y, 0]), s.feat_count), axis=None)
        else:
            s_vector = np.concatenate((np.array([s.x, s.y, self.feat_arr.index(s.feature)+1]), s.feat_count), axis=None)

        action_vector = np.zeros(len(self.solver._get_actions()))
        action_vector[a] = 1

        if not s_prime.feature:
            s_prime_vector = np.concatenate((np.array([s_prime.x, s_prime.y, 0]), s_prime.feat_count), axis=None)
        else:
            s_prime_vector = np.concatenate((np.array([s_prime.x, s_prime.y, self.feat_arr.index(s_prime.feature)+1]), s_prime.feat_count), axis=None)

        input_vector = np.concatenate((s_vector, action_vector, s_prime_vector), axis=None)

        if input_vector not in self.feat_map:
            self.feat_map = np.vstack([self.feat_map, input_vector])

        return input_vector

    def Q_learning(self):
        '''
        Compute the optimal policy pi_hat given the reward function R_hat outputed by the NN

        Inputs:
            None       

        Outputs:
            Q           A dictionary contains key=state, value = [action1, action2, action3]
        '''
        Q = defaultdict(lambda: np.zeros(self.solver.env.num_actions))

        episode = 200

        for i in range(episode):
            # if i % 10 == 0: print(f'Starting episode {i}')

            state = self.solver.env._reset()
            finished = self.solver.env.finished

            if self.lr > 0.05: self.lr *= 0.95
            if self.epsilon > 0.1: self.epsilon *= 0.95

            num_actions = 0

            while not finished and num_actions < 100:
                if random.uniform(0,1) < self.epsilon:
                    action = random.choice(self.solver.env.actions)
                else:
                    action = np.argmax(Q[state.__str__()])

                # change the CellState Object to a tensor, so we could pass it through the neural network
                _, next_state = self.T(state, action)[0]
                input_vector = self.get_input_vector(state, action, next_state)

                _, reward, finished, _ = self.solver.env.step(action, [self.fcnn, input_vector])

                old_Q_value = Q[state.__str__()][action]
                next_max = np.max(Q[next_state.__str__()])

                new_Q_value = (1. - self.lr) * old_Q_value + self.lr * (reward + GAMMA * next_max) # No need for the TD term here: # - old_Q_value)
                Q[state.__str__()][action] = new_Q_value

                state = next_state
                num_actions += 1

        return Q

    def policy_from_q(self, q_function):
        '''
        Input:
        q_function  dictionary[state] = [values for each action]

        Output:
        policy      dictionary[state] = [softmax probability dist]
        '''
        pi = defaultdict(lambda: np.zeros(self.solver.env.num_actions))

        for state in q_function:

            qs = q_function[state]
            softmax = np.log(np.sum(np.exp(qs))) # Approx V(s)

            for a in range(qs.shape[0]):
                pi[state][a] = np.exp(qs[a] - softmax) # softmax >= qs[a], and as qs[a]->softmax, exponent->0 and e^0=1.

        return pi

    def policy_propagation(self, pi_hat, start_state, horizon=10):
        '''
        Input:
        pi          Policy function: dictionary[state] = [probability distribution over actions]
        horizon     Number of steps to assume an agent can take in the environment

        Output:
        svf_sum   State visitation frequencies dictionary[state] = expected visitation amount
        '''
        
        svf_sum = defaultdict(lambda: 0)
        
        svf_i = defaultdict(lambda: 0)
        svf_i[start_state.__str__()] = 1

        for i in range(horizon):

            svf_iplus1 = defaultdict(lambda: 0)
            # svf_i[goal_state] = 0 # TODO: Need to set all terminal states to 0 in svf_i

            for state in svf_i:

                # state is CellObject.__str__()
                # change it back to CellObject
                coord, feat, count = state.split(',')
                coord = coord[coord.index('(')+1:coord.index(')')]
                feat = feat[feat.index(':')+1:]
                count = count[count.index('[')+1:count.index(']')]

                state_obj = CellState(int(coord.split()[0]), int(coord.split()[1]), feat, np.array([int(c) for c in count.split()]))

                for action in self.solver._get_actions():
                    next_states = self.T(state_obj, action)

                    for prob, next_state in next_states:
                        svf_iplus1[next_state.__str__()] += prob * pi_hat[state][action] * svf_i[state] # Pretty sure Algo3 in the paper is written wrong, this is the corrected version

                svf_sum[state] += svf_i[state]

            svf_i = svf_iplus1

        for state in svf_i:
            svf_sum[state] += svf_i[state]            

        return {k:v/len(svf_sum.keys()) for k, v in svf_sum.items()} # TODO: double check to see if we should normalize it with the number of states

    def deep_maxent_irl(self):
        '''
        Maximum Entropy Deep IRL
        This function should either return you with the extra states you need to add to your feat_map, or a solution

        Inputs:
            None

        Output:
            R_hat           a NX1 vector that is the final reward function learned from IRL
        '''

        # compute the demontration state visitation frequency
        demo_svf = self.demo_trajs_svf()

        for i in range(N_ITERS):
            print(f'Training NN Iteration {i}\n')

            # compute optimal policy given the current Neural Network
            # Q_learning() isn't returning the policy, it's returning the Q function, but in this case close enough!
            q_function = self.Q_learning()
            pi_hat = self.policy_from_q(q_function)
            print(f'Optimal policy given neural network: {pi_hat}\n')

            # compute the expected state visitation frequency
            start_state = self.solver._get_start_state()
            exp_svf = self.policy_propagation(pi_hat, start_state)

            # Loss  L_{D} = log(pi) * demo_svf_with_actions

            # perform backpropagation on loss
            # .train() is to put the Pytorch model in training mode
            # calling train() method of Pytorch model is required for the model parameters to be updated during backpropagation
            self.fcnn.model.train()
            loss = self.fcnn.backward(demo_svf, exp_svf)

            print("--------------------------------------------\n")

        # R_hat = self.fcnn.forward(self.feat_map)
        # print(R_hat)
        pi_hat = self.Q_learning()
        print(f'\nOptimal policy given neural network: {pi_hat}\n')

        self.solver._set_policy(pi_hat)   
        _, _, final_svf = self.solver.policy_rollout(max_steps=TIMESTAMP)    
        print(f'Final svf {final_svf}\n') 

        R_hat = []

        return R_hat
