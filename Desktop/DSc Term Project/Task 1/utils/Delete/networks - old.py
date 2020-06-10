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
HIDDEN_SIZE_DEFAULT = 500
BATCH_SIZE_DEFAULT = 128
FULLY_MODULATED_DEFAULT = True
ACTIVATION_DEFAULT = torch.tanh
PARAMETER_INIT_FUNCTION_DEFAULT = nn.init.xavier_normal_ # for W and alpha
ETA_INIT_VALUE_DEFAULT = 0.1
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
        fully_modulated = params.get('fully_modulated', FULLY_MODULATED_DEFAULT)  # not needed
        parameter_init_function = params.get('parameter_init_function', PARAMETER_INIT_FUNCTION_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)      # not needed
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        # ------------ trainable parameters - note the 'requires_grad' parameter ------------
        # fixed: w (non-plastic) - multiply by prev and add to current, then activ
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size,hidden_size),requires_grad=True)
        parameter_init_function(self.W)

        # ------------ layers: - simple 2 layer neural network ------------
        # input to hidden - use He initialisation for the weights
        self.fc_i2h = torch.nn.Linear(input_size, hidden_size)
        nn.init.xavier_normal_(self.fc_i2h.weight)

        # hidden to output
        self.fc_h2o = torch.nn.Linear(hidden_size, output_size)
        nn.init.xavier_normal_(self.fc_h2o.weight)

        # hidden to v - for A2C
        self.fc_h2v = torch.nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.fc_h2v.weight)


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
        hidden = self.fc_i2h(inputs) + torch.matmul(prev, self.W)
        hidden = activation(hidden)

        # output: - pure linear -> will be softmaxed outside
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
        print("Correct NM!")

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        parameter_init_function = params.get('parameter_init_function', PARAMETER_INIT_FUNCTION_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(input_size, hidden_size)
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.eta = torch.nn.Parameter((.1 * torch.ones(1)), requires_grad=True)  # Everyone has the same eta
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.h2v = torch.nn.Linear(hidden_size, 1)

    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    def forward(self, inputs, cache):
        (hidden, hebb) = cache
        params = self.params
        
        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)

        # Here, the *rows* of w and hebb are the inputs weights to a single neuron
        # hidden = x, hactiv = y
        hactiv = self.activ(self.i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),hidden.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
        valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...

        # deltahebb has shape BS x hidden_size x hidden_size
        # Each row of hebb contain the input weights to a neuron
        deltahebb =  torch.bmm(hactiv.view(batch_size, hidden_size, 1), hidden.view(batch_size, 1, hidden_size)) # batched outer product...should it be other way round?
        hebb = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)

        hidden = hactiv
        
        cache = (hidden, hebb)
        return activout, valueout, cache
        
        

# ================= 3. Plastic, Simple Modulation: =================
class Plastic_SimpleModulated_RNN(nn.Module):
    def __init__(self, params):
        # ------------ init call ------------
        super(Plastic_SimpleModulated_RNN, self).__init__()
        self.params = params
        print("final pls")

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        NBDA = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        parameter_init_function = params.get('parameter_init_function', PARAMETER_INIT_FUNCTION_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(input_size, hidden_size)
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.h2DA = torch.nn.Linear(hidden_size, NBDA)
    
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.h2v = torch.nn.Linear(hidden_size, 1)


    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    def forward(self, inputs, cache):
        (hidden, hebb) = cache
        params = self.params
        
        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)
        NBDA = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        neuromod_activation = params.get('neuromod_activation', NEUROMOD_ACTIVATION_DEFAULT)
        fully_modulated = params.get('fully_modulated', FULLY_MODULATED_DEFAULT)


        hactiv = self.activ(self.i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                        hidden.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
        valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...

        # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
        DAout = F.tanh(self.h2DA(hactiv)) 

        # deltahebb has shape BS x hidden_size x hidden_size
        # Each row of hebb contain the input weights to a neuron
        deltahebb =  torch.bmm(hactiv.view(batch_size, hidden_size, 1), hidden.view(batch_size, 1, hidden_size)) # batched outer product...should it be other way round?
        hebb = torch.clamp(hebb + DAout.view(batch_size, 1, 1) * deltahebb, min=-1.0, max=1.0)

        hidden = hactiv
        cache = (hidden, hebb)

        return activout, valueout, cache 



# ================= 4. Plastic, Retroactive Modulation: =================
class Plastic_RetroactiveModulated_RNN(nn.Module):
    def __init__(self, params):
        # ------------ init call ------------
        super(Plastic_RetroactiveModulated_RNN, self).__init__()
        self.params = params
        print("Corrected !!")

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        NBDA = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        parameter_init_function = params.get('parameter_init_function', PARAMETER_INIT_FUNCTION_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        self.params = params
        self.activ = F.tanh
        self.i2h = torch.nn.Linear(input_size, hidden_size)
        self.w =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.alpha =  torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        self.eta = torch.nn.Parameter((.1 * torch.ones(1)), requires_grad=True)  # Everyone has the same eta (only for the non-modulated part, if any!)
        self.etaet = torch.nn.Parameter((.1 * torch.ones(1)), requires_grad=True)  # Everyone has the same etaet
        self.h2DA = torch.nn.Linear(hidden_size, NBDA)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.h2v = torch.nn.Linear(hidden_size, 1)
        

    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    # E_t is eligibility trace, plastic_weights is M*E_t
    def forward(self, inputs, cache):
        (hidden, hebb, et, pw) = cache
        params = self.params
        
        # ------------ extract parameters ------------
        batch_size = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)
        NBDA = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        neuromod_activation = params.get('neuromod_activation', NEUROMOD_ACTIVATION_DEFAULT)
        fully_modulated = params.get('fully_modulated', FULLY_MODULATED_DEFAULT)

        hactiv = self.activ(self.i2h(inputs).view(batch_size, hidden_size, 1) + torch.matmul((self.w + torch.mul(self.alpha, pw)), hidden.view(batch_size, hidden_size, 1))).view(batch_size, hidden_size)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
        valueout = self.h2v(hactiv)
        DAout = F.tanh(self.h2DA(hactiv))

        deltahebb = torch.bmm(hactiv.view(batch_size, hidden_size, 1), hidden.view(batch_size, 1, hidden_size)) # batched outer product...should it be other way round?

        # Hard clamp
        deltapw = DAout.view(batch_size,1,1) * et
        pw = torch.clamp(pw + deltapw, min=-1.0, max=1.0)

        # Updating the eligibility trace - always a simple decay term.
        # Note that self.etaet != self.eta (which is used for hebb, i.e. the non-modulated part)
        deltaet = deltahebb
        et = (1 - self.etaet) * et + self.etaet *  deltaet

        hidden = hactiv
        cache = (hidden, hebb, et, pw)
        return activout, valueout, cache

