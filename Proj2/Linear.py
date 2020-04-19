import math
import torch
from Module import Module

class Linear(Module):
    """Class representing a fully connected linear layer in a neural network."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.params = []
        tmp = math.sqrt(in_features) # For weight initialization.
        self.weight = torch.empty(size=(out_features, in_features)).uniform_(-1/tmp,1/tmp)
        self.dw = torch.zeros(size=(out_features, in_features))
        self.params.append((self.weight, self.dw))
        if bias:
            self.bias = torch.empty(out_features).normal_(mean=mean, std=std)
            self.db = torch.zeros(out_features)
            self.params.append((self.bias, self.db))
        else:
            self.bias = None
            self.db = None
            
    def forward(self, input):
        """
        Forwards the input data by applying a linear transformation on it.
        
        Args:
        input -- tensor of size (N, *, in_features)
        
        Returns:
        output -- tensor of size (N, *, out_features)
        """
        
        assert(input.size(-1) == self.in_features)
        
        
        # Required information for the backward pass.
        # We clone to ensure that input won't be modified by the user before
        # calling backward.
        self.input = input.clone()
        
        output = input @ self.weight.T
        if self.bias is not None:
            output += self.bias
        return output
        
    __call__ = forward
        
    def backward(self, grad_output):
        """
        Computes the gradient w.r.t. the input of the layer
        given the gradient w.r.t. to the output of the layer.
        Also computes and updates the gradient w.r.t.
        the parameters of the layer.
        
        Args:
        grad_output -- tensor of size (N, *, out_features)
        
        Returns 
        grad_input -- tensor of size (N, * , in_features)
        """

        assert(grad_output.size(-1) == self.out_features)
        
        grad_input = grad_output @ self.weight
        
        if self.bias is not None:
            self.db += grad_output.sum(axis=0)
        self.dw += grad_output.T @  self.input
        
        return grad_input
       
    def zero_grad(self):
        """
        Sets the gradient w.r.t. the parametes to zero.
        """
        
        self.dw = torch.zeros(size=(self.out_features, self.in_features))
        
        if self.bias is not None:
            self.db = torch.zeros(self.out_features)
        return
    
    def param(self):
        """
        Returns the parameters of the layer i.e. its weight and
        bias along with their gradient.
        
        Returns:
        params -- a list of pairs, each composed of a parameter tensor, 
        and a gradient tensor of same size.
        """
        
        # We just return a copy as we don't want the user
        # to be able to change the params of the model through this method.
        params = self.params.copy() 
        return params 
    

    def update_params(self, step_size):
        """
        Update the parameters of the linear layer by going 
        in the opposite direction of the gradient.
        
        Args:
        step_size -- the size of an update step
        """
        
        self.weight -= step_size * self.dw
        if self.bias is not None:
            self.bias -= step_size * self.db
        return
    