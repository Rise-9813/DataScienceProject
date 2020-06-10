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
        print("Corrected!")

        # extract parameters
        input_size = params.get('input_size')
        output_size = params.get('output_size')
        hidden_size = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)
        eta_init_value = params.get('eta_init_value', ETA_INIT_VALUE_DEFAULT)
        parameter_init_function = params.get('parameter_init_function', PARAMETER_INIT_FUNCTION_DEFAULT)
        assert (input_size is not None) and (output_size is not None), "'input_size and output_size' must be specified!"

        # ------------ trainable parameters - note the 'requires_grad' parameter ------------
        # fixed: w (non-plastic) - multiply by prev and add to current, then activ
        #self.W = torch.nn.Parameter(torch.Tensor(hidden_size,hidden_size),requires_grad=True)
        self.W = torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        #parameter_init_function(self.W)

        # plastic: alpha - scale parameter determining maximum magnitude of the plastic component
        #self.alpha = torch.nn.Parameter(torch.Tensor(hidden_size,hidden_size),requires_grad=True)
        self.alpha = torch.nn.Parameter((.01 * torch.t(torch.rand(hidden_size, hidden_size))), requires_grad=True)
        #parameter_init_function(self.alpha)

        # plastic: eta - intra life learning rate determining how fast new info is incorporated ino the plastic component -> all have same eta
        self.eta = torch.nn.Parameter((eta_init_value * torch.ones(1)), requires_grad=True)
        #nn.init.constant_(self.eta, eta_init_value)

        # ------------ layers: - simple 2 layer neural network ------------
        # input to hidden - use He initialisation for the weights
        self.fc_i2h = torch.nn.Linear(input_size, hidden_size)
        #nn.init.xavier_normal_(self.fc_i2h.weight)

        # hidden to output
        self.fc_h2o = torch.nn.Linear(hidden_size, output_size)
        #nn.init.xavier_normal_(self.fc_h2o.weight)

        # hidden to v - for A2C
        self.fc_h2v = torch.nn.Linear(hidden_size, 1)
        #nn.init.xavier_normal_(self.fc_h2v.weight)


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
        hidden = self.fc_i2h(inputs).view(batch_size, hidden_size, 1)
        hidden += torch.matmul(self.W + (self.alpha * hebb),prev.view(batch_size, hidden_size, 1))
        hidden = activation(hidden).view(batch_size, hidden_size)

        # output: - pure linear -> will be softmaxed outside
        a_out = self.fc_h2o(hidden) # action
        v_out = self.fc_h2v(hidden) # for A2C

        # ------------ hebb update ------------
        # hebb = clip(hebb + eta * prev.current)
        delta_hebb = torch.bmm(hidden.view(batch_size, hidden_size, 1), prev.view(batch_size, 1, hidden_size))
        hebb = torch.clamp(hebb + self.eta * delta_hebb, min = -1.0, max = 1.0)

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
        self.eta = torch.nn.Parameter((.1 * torch.ones(1)), requires_grad=True)  # Everyone has the same eta (only for the non-modulated part, if any!)
        self.etaet = torch.nn.Parameter((.1 * torch.ones(1)), requires_grad=True)  # Everyone has the same etaet
        self.h2DA = torch.nn.Linear(hidden_size, NBDA)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        self.h2v = torch.nn.Linear(hidden_size, 1)


    # forward pass - prev is from the previous hidden layer, hebb is plastic content
    def forward(self, inputs, cache):
        (hidden, hebb) = cache
        params = self.params
        
        # ------------ extract parameters ------------
        BATCHSIZE = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        HS = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)
        NBDA = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        neuromod_activation = params.get('neuromod_activation', NEUROMOD_ACTIVATION_DEFAULT)
        fully_modulated = params.get('fully_modulated', FULLY_MODULATED_DEFAULT)


        hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, hebb)),
                        hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed by the calling program
        valueout = self.h2v(hactiv)

        # Now computing the Hebbian updates...

        # With batching, DAout is a matrix of size BS x 1 (Really BS x NBDA, but we assume NBDA=1 for now in the deltahebb multiplication below)
        DAout = F.tanh(self.h2DA(hactiv)) 

        # deltahebb has shape BS x HS x HS
        # Each row of hebb contain the input weights to a neuron
        deltahebb =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?


        hebb1 = torch.clamp(hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=-1.0, max=1.0)
        if fully_modulated == False:
            # Non-modulated part
            hebb2 = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
        # Soft Clamp (note that it's different from just putting a tanh on top of a freely varying value):
        #hebb1 = torch.clamp( hebb +  torch.clamp(DAout.view(BATCHSIZE, 1, 1) * deltahebb, min=0.0) * (1 - hebb) +
        #        torch.clamp(DAout.view(BATCHSIZE, 1, 1)  * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
        #hebb2 = torch.clamp( hebb +  torch.clamp(self.eta * deltahebb, min=0.0) * (1 - hebb) +  torch.clamp(self.eta * deltahebb, max=0.0) * (hebb + 1) , min=-1.0, max=1.0)
        # Purely additive, no clamping. This will almost certainly diverge, don't use it!
        #hebb1 = hebb + DAout.view(BATCHSIZE, 1, 1) * deltahebb
        #hebb2 = hebb + self.eta * deltahebb

        if fully_modulated:
            hebb = hebb1
        else:
            # Combine the modulated and non-modulated part
            hebb = torch.cat( (hebb1[:, :HS//2, :], hebb2[:,  HS // 2:, :]), dim=1) # Maybe along dim=2 instead?...

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
        BATCHSIZE = params.get('batch_size', BATCH_SIZE_DEFAULT)     # not needed
        HS = params.get('hidden_size', HIDDEN_SIZE_DEFAULT)  # not needed
        activation = params.get('activation', ACTIVATION_DEFAULT)
        NBDA = params.get('neuromod_neurons', NEUROMOD_NEURONS_DEFAULT)
        neuromod_activation = params.get('neuromod_activation', NEUROMOD_ACTIVATION_DEFAULT)
        fully_modulated = params.get('fully_modulated', FULLY_MODULATED_DEFAULT)

        hactiv = self.activ(self.i2h(inputs).view(BATCHSIZE, HS, 1) + torch.matmul((self.w + torch.mul(self.alpha, pw)), hidden.view(BATCHSIZE, HS, 1))).view(BATCHSIZE, HS)
        activout = self.h2o(hactiv)  # Pure linear, raw scores - will be softmaxed later
        valueout = self.h2v(hactiv)
        DAout = F.tanh(self.h2DA(hactiv))

        deltahebb = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product...should it be other way round?

        # Hard clamp
        deltapw = DAout.view(BATCHSIZE,1,1) * et
        pw1 = torch.clamp(pw + deltapw, min=-1.0, max=1.0)

        # Should we have a fully neuromodulated network, or only half?
        if fully_modulated:
            pw = pw1
        else:
            hebb = torch.clamp(hebb + self.eta * deltahebb, min=-1.0, max=1.0)
            pw = torch.cat( (hebb[:, :HS//2, :], pw1[:,  HS // 2:, :]), dim=1) # Maybe along dim=2 instead?...


        # Updating the eligibility trace - always a simple decay term.
        # Note that self.etaet != self.eta (which is used for hebb, i.e. the non-modulated part)
        deltaet = deltahebb
        et = (1 - self.etaet) * et + self.etaet *  deltaet

        hidden = hactiv
        cache = (hidden, hebb, et, pw)
        return activout, valueout, cache

