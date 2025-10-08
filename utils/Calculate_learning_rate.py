import numpy as np
import copy
import torch


class LearningRateUpdater():

    def __init__(self, model_parameters):

        self.current_round = 0
        self.mean = self.initializeMean(model_parameters)  # Adaptive-FedAVG mean initialization
        self.previous_mean = copy.deepcopy(self.mean)
        self.previous_mean_loss = 0
        self.mean_loss = 0  # Adaptive-FedAVG loss-based mean initialization
        self.variance = 0  # Adaptive-FedAVG variance initialization
        self.previous_variance = 0
        self.LRcoeff = 0
        self.c = 0  # FedREG coefficient
        self.model_parameters = model_parameters

    #TODO：rewrite initializeMean function
    def initializeMean(self,model_parameters):
        '''
            Initialize the mean vector of parameters to 0
        '''
        with torch.no_grad():
            a = np.array([])
            #for i in model_parameters:
            for i in model_parameters.values():
                #for cifar
                if i.dim() > 0:
                    for j in i:
                        a = np.append(a, j.clone().cpu())
                else:
                    a = np.append(a,i.clone().cpu().item())
            a = np.zeros(len(a))

        return a

    def adaptiveLR(self, newParameters):
        '''

        :param self:
        :param newParameters:
        :return: Learning rate
        '''
        beta1=0.5
        beta2=0.5
        beta3=0.5

        initialLR = 0.06 


        # Create the array containing newParameters
        newParameters_arr = np.array([])
        for i in newParameters.values():
            if i.dim()>0:
                for j in i:
                    newParameters_arr = np.append(newParameters_arr, j.clone().cpu().data.numpy())
            else:
                newParameters_arr = np.append(newParameters_arr, i.clone().cpu().data.numpy())

        # EMA on the mean
        self.mean = self.previous_mean * beta1 + (1 - beta1) * newParameters_arr

        # Initialization Bias correction
        self.mean = self.mean / (1 - pow(beta1, self.current_round + 1))

        # EMA on the Variance
        self.variance = self.previous_variance * beta2 + (1 - beta2) * np.mean(
            (newParameters_arr - self.previous_mean) * (newParameters_arr - self.previous_mean))

        self.previous_mean = copy.deepcopy(self.mean)

        temp = copy.deepcopy(self.previous_variance)
        self.previous_variance = copy.deepcopy(self.variance)
        # Initialization Bias correction
        self.variance = self.variance / (1 - pow(beta2, self.current_round + 1))

        if self.current_round < 2:
            r = 1
        else:
            r = np.abs(self.variance / (temp / (1 - pow(beta2, self.current_round))))

        self.LRcoeff = self.LRcoeff * beta3 + (1 - beta3) * r

        coeff = self.LRcoeff / (1 - pow(beta3, self.current_round + 1))

        ### SERVER-SIDE LEARNING RATE SCHEDULER

        # No Decay
        coeff = min(initialLR, initialLR*coeff)
        #DEBUG
        print("coeff:{}".format(coeff))

        # Decay of 1/t TIME BASED DECAY as in the convergence analysis of FedAVG
        # coeff = min(initialLR, (initialLR * coeff) / (self.current_round + 1))

        # Decay of 0.99
        # coeff = min(settings.initialLR, settings.initialLR*coeff*math.pow(0.99, self.current_round))

        return coeff

    def set_current_round(self):
        self.current_round+=1

    def set_current_round_for_cifar(self):
        self.current_round+=10