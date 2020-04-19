import torch
from Module import Module

class ReLU(Module):
    """Class representing the rectified linear unit activation function."""
    
    def forward(self, input) :
        """
        Apllies the relu function to the input.
        
        Args:
        input -- tensor of size (N, *)
        
        Return:
        output -- tensor of same size as input
        """
        
        # The input is needed when computing the backward pass.
        self.input = input.clone()
        
        output = input.clamp(min=0)
        return output
    
    __call__ = forward
    
    def backward(self, grad_output):
        """
        Given the gradient w.r.t. to the output of the activation,
        computes the gradient w.r.t. to the input of the activation.
        
        Args:
        grad_output -- tensor of same size as self.input
        
        Returns:
        grad_input -- tensor of same size as self.input
        """
        
        assert(self.input is not None)
        assert(grad_output.size() == self.input.size())
        
        grad_input = grad_output.clone()
        grad_input[self.input < 0] = 0
        return grad_input

class Tanh(Module):
    """Class representing the hyperbolic tangent activation function."""
    
    def forward(self, input):
        """
        Applies the hyperbolic tangent to the input.
        
        Args:
        input -- tensor of size (N, *)
        
        Returns:
        output -- tensor of same size as input
        """
        
        # The input is needed when computing the backward pass.
        self.input = input.clone()
        
        output = torch.tanh(input)
        return output
    
    __call__ = forward
    
    def backward (self, grad_output):
        """
        Given the gradient w.r.t. to the output of the activation,
        computes the gradient w.r.t. to the input of the activation.
        
        Args:
        grad_output -- tensor of same size as self.input
        
        Returns:
        grad_input -- tensor of same size as self.input
        """
        
        grad_input = 1 - (self.input.tanh() ** 2) 
        grad_input = grad_output * grad_input
        return grad_input