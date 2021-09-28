from typing import OrderedDict
import numpy as np
import torch
import random
from torch import nn
from torch._C import dtype
from torch.nn import *
from collections import OrderedDict, defaultdict, Counter
from pref_solver import GAMMA, LR, N_ITERS, TIMESTAMP, N_FEAT
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
        input = torch.tensor(state_tensor, dtype=torch.float, requires_grad=True)

        reward = self.model(input)

        return reward

    def backward(self, demo_svf, exp_svf):

        # a = torch.tensor(np.array([1, 1, 1, 1, 1, 1, 0, 0]), dtype=float)
        # b = torch.tensor(np.array([1, 1, 5, 0, 0, 0, 0, 0]))
        # c = torch.tensor(np.array([1, 0, 0, 0, 0, 0, 1, 5]))

        # print(f'Loss for (b, a): {self.loss_fn(b, a)}')
        # print(f'Loss for (c, a): {self.loss_fn(c, a)}')

        print(f'demo_svf:{demo_svf}')
        print('\n')
        print(f'exp_svf: {exp_svf}')
        print('\n')

        label = torch.tensor(np.array([v for v in demo_svf.values()]), dtype=float)
        print(label)
        print('\n')

        pred = np.array([])
        for state in demo_svf.keys():
            if state in exp_svf.keys():
                pred = np.append(pred, exp_svf[state])
            else:
                pred = np.append(pred, TIMESTAMP+1) # instead of 0, to avoid edge cases
        pred = torch.tensor(pred)
        print(pred)
        print('\n')

        loss = self.loss_fn(pred, label)
        loss.requires_grad = True
        print(f'Loss: {loss}')
        print('\n')

        # Zero the gradiants accumulated from the previous steps
        self.optimizer.zero_grad()

        # Perform backpropagation
        loss.backward()

        # Update model parameters
        self.optimizer.step()

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

    def state_to_tensor(self, s):
        '''
        Change the states [0, 0, None, 0, 0] to a tensor that can be passed through neural network

        Input;
            s       a state

        Output:
            state_vector    a tensor that could be used as an input to the neural network
        '''

        if not s.feature:
            state_vector = np.concatenate((np.array([s.x, s.y, 0]), s.feat_count), axis=None)
        else:
            state_vector = np.concatenate((np.array([s.x, s.y, self.feat_arr.index(s.feature)+1]), s.feat_count), axis=None)

        if state_vector not in self.feat_map:
            self.feat_map = np.vstack([self.feat_map, state_vector])

        return state_vector

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
                input_tensor = self.state_to_tensor(state)
                next_state, reward, finished, info = self.solver.env.step(action, [self.fcnn, input_tensor])

                old_Q_value = Q[state.__str__()][action]
                next_max = np.max(Q[next_state.__str__()])

                new_Q_value = (1. - self.lr) * old_Q_value + self.lr * (reward + GAMMA * next_max - old_Q_value)
                Q[state.__str__()][action] = new_Q_value

                state = next_state
                num_actions += 1

        return Q

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

    def exp_policy_svf(self, pi_hat):
        '''
        Compute state visitation frequencies from policy pi_hat

        Inputs:
            pi_hat      policy given current reward function R_hat

        Outputs:
            exp_svf     Expected state visitation frequencies of pi_hat
        '''
        # first set the current policy to be our pi_hat, before rolling out the policy
        self.solver._set_policy(pi_hat)

        # roll out the policy n steps into the future
        _, _, exp_svf = self.solver.policy_rollout(max_steps=TIMESTAMP)

        # {CellStateObject1: 1, CellStateObject2: 1, ...}
        return exp_svf

    def deep_maxent_irl(self):
        '''
        Maximum Entropy Deep IRL
        This function should either return you with the extra states you need to add to your feat_map, or a solution

        Inputs:
            None

        Output:
            R_hat           a NX1 vector that is the final reward function learned from IRL
        '''

        demo_svf = self.demo_trajs_svf()

        for i in range(N_ITERS):
            print(f'Training NN Iteration {i}')

            # compute optimal policy given the current Neural Network
            pi_hat = self.Q_learning()
            print(f'\nOptimal policy given neural network: {pi_hat}\n')

            # compute the expected svf of the current optimal policy
            exp_svf = self.exp_policy_svf(pi_hat)


            # make demo_svf and exp_svf contain the same states
            # for state in exp_svf.keys():
            #     if state not in demo_svf.keys():
            #         demo_svf[state] = 0

            # ordered_exp_svf = OrderedDict()
            # for state in demo_svf.keys():
            #     if state not in exp_svf.keys():
            #         ordered_exp_svf[state] = 0
            #     else:
            #         ordered_exp_svf[state] = exp_svf[state]

            # for k, v in exp_svf.items():
            #     print(f'State: {k}, svf: {v}')
            # print('\n')
            # for k, v in demo_svf.items():
            #     print(f'State: {k}, svf: {v}')
            # print('\n')

            # perform backpropagation on loss
            # .train() is to put the Pytorch model in training mode
            # calling train() method of Pytorch model is required for the model parameters to be updated during backpropagation
            self.fcnn.model.train()
            loss = self.fcnn.backward(demo_svf, exp_svf)

            if loss < 0.05: break

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
