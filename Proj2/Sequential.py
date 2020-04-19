from Module import Module

class Sequential(Module):
    """Class reprensing a neural network composed of several modules"""
    
    def __init__(self, *modules):
        self.modules = list(modules)
        
    
    def forward(self, input):
        """
        Computes the output of the neural network given some input.
        
        Args:
        input -- tensor of size (N, *, i), i = in_features of the first layer of the nn.
        
        Returns:
        output -- tensor of size (N, *, o), o = out_features of the last layer of the nn.
        """
        output = input
        for module in self.modules:
            output = module(output)
        return output

    __call__ = forward
    
    def backward(self, grad_output):
        """
        Performs whole backward pass of the neural networks.
        
        Args:
        grad_output -- gradient of the loss w.r.t. the output of the nn.
        
        Returns:
        grad_input -- gradient of the loss w.r.t. to the input of the nn.
        """
        grad_input = grad_output
        self.modules.reverse()
        for module in self.modules:
            grad_input = module.backward(grad_input)
        self.modules.reverse()
        return grad_input
    
    def zero_grad(self):
        """
        Sets the gradient w.r.t. the parametes  of the nn. to zero.
        """
        for module in self.modules:
            module.zero_grad() 
        return
    
    def param(self):
        """
        Returns the parameters of the nn along with their gradient.
        
        Returns:
        params -- a list of pairs, each composed of a parameter tensor, 
        and a gradient tensor of same size.
        """
        params = []
        for module in self.modules:
            params.append(module.param())
        return params
    

    def update_params(self, step_size):
        """
        Update the parameters of the nn going 
        in the opposite direction of the gradient.
        
        Args:
        step_size -- the size of an update step
        """
        
        for module in self.modules:
            module.update_params(step_size) 
        return