# The 4 different types of recurrent neural networks used in the analysis are given here:
#       1. Non-Plastic, Non-Modulated
#       2. Plastic, Non-Modulated
#       3. Plastic, Simple Neuromodulation
#       4. Plastic, Retroactive Neuromodulation

# import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# constants
HIDDEN_SIZE_DEFAULT = 200
BATCH_SIZE_DEFAULT = 10
ACTIVATION_DEFAULT = torch.tanh
ETA_INIT_VALUE_DEFAULT = 0.01
NEUROMOD_NEURONS_DEFAULT = 1
NEUROMOD_ACTIVATION_DEFAULT = torch.tanh


# ================= 1. Non-Plastic, Non-Modulated: =================
class NonPlastic_NonModulated_RNN(nn.Module):
    def __init__(self, params):
        # ------------ init call ------------
        super(NonPlastic_NonModulated_RNN, self).__init__()
        self.params = params

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)                 # not needed
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        # ------------ trainable parameters - note the 'requires_grad' parameter ------------
        # fixed: w (non-plastic) - multiply by prev and add to current, then activ
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)

        # ------------ layers: - simple 2 layer neural network ------------
        # input to hidden - use He initialisation for the weights
        self.fc_i2h = torch.nn.Linear(input_size, hidden_size)

        # hidden to output
        self.fc_h2o = torch.nn.Linear(hidden_size, output_size)

        # hidden to v - for A2C
        self.fc_h2v = torch.nn.Linear(hidden_size, 1)


    # forward pass - prev is from the previous hidden layer
    # we use cache so that all forward layers can have same interface
    def forward(self, inputs, cache):
        prev = cache[0]
        params = self.params

        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)

        # ------------ forward pass ------------
        # current + W*prev -> check sizes $$
        hidden = activation(self.fc_i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul(self.w,
                        prev.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)

        # output: - pure linear -> softmax outside
        a_out = self.fc_h2o(hidden) # action
        v_out = self.fc_h2v(hidden) # for A2C

        # ------------ return ------------
        # set this as prev for next and return
        prev = hidden

        cache = (prev,)
        return a_out, v_out, cache


# ================= 2. Plastic, Non-Modulated: =================
class Plastic_NonModulated_RNN(nn.Module):
    def __init__(self, params):
        # ------------ init call ------------
        super(Plastic_NonModulated_RNN, self).__init__()
        self.params = params

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        # ------------ trainable parameters - note the 'requires_grad' parameter ------------
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.eta = torch.nn.Parameter((eta_init_value * torch.ones(1)), requires_grad=True)  # Everyone has the same eta

        # ------------ layers: - simple 2 layer neural network ------------
        self.fc_i2h = torch.nn.Linear(input_size, hidden_size)
        self.fc_h2o = torch.nn.Linear(hidden_size, output_size)
        self.fc_h2v = torch.nn.Linear(hidden_size, 1)


    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    def forward(self, inputs, cache):
        (prev, hebb) = cache
        params = self.params

        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)

        # ------------ forward pass ------------
        # current + (W + (alpha*hebb)) * prev -> check sizes $$
        hidden = activation(self.fc_i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),prev.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)
        a_out = self.fc_h2o(hidden)  # Pure linear, raw scores - will be softmaxed by the calling program
        v_out = self.fc_h2v(hidden)

        # ------------ hebb update ------------
        # hebb = clip(hebb + eta * prev.current)
        delta_hebb =  torch.bmm(hidden.view(batch_size, hidden_size, 1), prev.view(batch_size, 1, hidden_size))
        hebb = torch.clamp(hebb + self.eta * delta_hebb, min=-1.0, max=1.0)

        # ------------ return ------------
        # set this as prev for next and return
        prev = hidden

        cache = (prev, hebb)
        return a_out, v_out, cache



# ================= 3. Plastic, Simple Modulation: =================
class Plastic_SimpleModulated_RNN(nn.Module):
    def __init__(self, params):
        # ------------ init call ------------
        super(Plastic_SimpleModulated_RNN, self).__init__()
        self.params = params

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        neuromod_neurons = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        # ------------ trainable parameters - note the 'requires_grad' parameter ------------
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)

        # ------------ layers: - simple 2 layer neural network ------------
        self.fc_h2mod = torch.nn.Linear(hidden_size, neuromod_neurons)
        self.fc_i2h = torch.nn.Linear(input_size, hidden_size)
        self.fc_h2o = torch.nn.Linear(hidden_size, output_size)
        self.fc_h2v = torch.nn.Linear(hidden_size, 1)


    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    def forward(self, inputs, cache):
        (prev, hebb) = cache
        params = self.params

        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)
        neuromod_neurons = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        neuromod_activation = params.get('neuromod_activation', NEUROMOD_ACTIVATION_DEFAULT)

        # ------------ forward pass ------------
        # current + (W + (alpha*hebb)) * prev -> check sizes $$
        hidden = activation(self.fc_i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                        prev.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)
        a_out = self.fc_h2o(hidden)  # Pure linear, raw scores - will be softmaxed by the calling program
        v_out = self.fc_h2v(hidden)
        M_out = neuromod_activation(self.fc_h2mod(hidden)) # for neuromodulation
        M_out = torch.mean(M_out, axis=1) # take mean of what the neuromod neurons are telling you

        # ------------ hebb update hebb = clip(hebb + [M or eta or mix] * prev.current) ------------
        # hebb = clip(hebb + eta * prev.current)
        delta_hebb = torch.bmm(hidden.view(batch_size, hidden_size, 1), prev.view(batch_size, 1, hidden_size))
        hebb = torch.clamp(hebb + M_out.view(batch_size, 1, 1) * delta_hebb, min=-1.0, max=1.0)

        # ------------ return ------------
        # set this as prev for next and return
        prev = hidden
        cache = (prev, hebb)

        return a_out, v_out, cache



# ================= 4. Plastic, Retroactive Modulation: =================
class Plastic_RetroactiveModulated_RNN(nn.Module):
    def __init__(self, params):
        # ------------ init call ------------
        super(Plastic_RetroactiveModulated_RNN, self).__init__()
        self.params = params

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        neuromod_neurons = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        # ------------ trainable parameters - note the 'requires_grad' parameter ------------
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.eta_et = torch.nn.Parameter((eta_init_value * torch.ones(1)), requires_grad=True)  # Everyone has the same eta_et

        # ------------ layers: - simple 2 layer neural network ------------
        self.fc_h2mod = torch.nn.Linear(hidden_size, neuromod_neurons)
        self.fc_i2h = torch.nn.Linear(input_size, hidden_size)
        self.fc_h2o = torch.nn.Linear(hidden_size, output_size)
        self.fc_h2v = torch.nn.Linear(hidden_size, 1)


    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    # E_t is eligibility trace, plastic_weights is M*E_t
    def forward(self, inputs, cache):
        (prev, hebb, E_t, plastic_weights) = cache
        params = self.params

        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)
        neuromod_neurons = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        neuromod_activation = params.get('neuromod_activation', NEUROMOD_ACTIVATION_DEFAULT)

        # ------------ forward pass ------------
        # current + (W + (alpha*plastic_weights)) * prev -> check sizes $$
        hidden = activation(self.fc_i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul((self.w + torch.mul(self.alpha, plastic_weights)), prev.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)
        a_out = self.fc_h2o(hidden)  # Pure linear, raw scores - will be softmaxed later
        v_out = self.fc_h2v(hidden)
        M_out = neuromod_activation(self.fc_h2mod(hidden))
        M_out = torch.mean(M_out, axis=1) # take mean of what the neuromod neurons are telling you

        # ------------ hebb update -> plastic_weights = clip(plastic_weights + M * E_t) ------------
        deltapw = M_out.view(batch_size,1,1) * E_t
        plastic_weights = torch.clamp(plastic_weights + deltapw, min=-1.0, max=1.0)

        # ------------ eligibility trace E_t update -> (1-eta_E)E_t + (eta_E)*[deltahebb = x(t-1)x(t)] ------------
        delta_et = torch.bmm(hidden.view(batch_size, hidden_size, 1), prev.view(batch_size, 1, hidden_size))
        E_t = (1 - self.eta_et) * E_t + self.eta_et *  delta_et

        prev = hidden
        cache = (prev, hebb, E_t, plastic_weights)
        return a_out, v_out, cache
