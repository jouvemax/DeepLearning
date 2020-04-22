class Module(object) :
    """Base class for all neural network modules."""

    def forward (self , *input) :
        raise NotImplementedError
        
    def backward (self , *gradwrtoutput):
        raise NotImplementedError 

    def update_params(self, step_size):
        return
    
    def zero_grad(self):
        return
    
    def param(self):
        return []
    