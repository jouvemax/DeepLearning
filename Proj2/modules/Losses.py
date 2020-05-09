from modules import Module

class LossMSE(Module):
    """Class representing mean square error loss."""
    
    def forward(self, input, target):
        """
        Computes the mean square error loss between 
        the input tensor and the target tensor.

        Args:
        input -- tensor of size (N, D)
        target -- tensor of size (N, D)

        Returns:
        loss -- mse loss between input and target
        """
    
        assert(input.size() == target.size())

        loss = (target-input).pow(2).mean()
        return loss
    
    __call__ = forward
    
    def backward(self, input, target):
        """
        Computes the gradient of the cross entropy loss w.r.t. 
        the input tensor.
        
        Args:
        input -- tensor of size (N, 1)
        target -- tensor of size (N, 1)
    
        Returns:
        grad -- tensor of same size as input
        """
        
        assert(input.size() == target.size())
        
        N = input.numel()
        grad = input-target
        
        assert(grad.size() == input.size())
        return 2/N * grad
    
class LossCrossEntropy(Module):
    """Class representing the cross entropy loss."""
 
    def forward(self, input, target):
        """
        Computes the cross entropy loss given 
        the input tensor and the target tensor.

        Args:
        input -- tensor of size (N, D)
        target -- tensor of size (N, 1)

        Returns:
        loss -- cross entropy loss between input and target
        """
    
        assert(input.size(0) == target.size(0))

        N = input.size(0)
        tmp1 = input.exp().sum(axis=1)
        tmp2 = input[:,target].diag().exp()
        loss = -1/N * (tmp2/tmp1).log().sum()
        return loss
    
    __call__ = forward
    
    def backward(self, input, target):
        """
        Computes the gradient of the cross entropy loss w.r.t. 
        the input tensor.
        
        Args:
        input -- tensor of size (N, D)
        target -- tensor of size (N, 1)
    
        Returns:
        grad -- tensor of same size as input
        """
        
        assert(input.size(0) == target.size(0))
        N = input.size(0)
        exp = input.exp()
        sum = exp.sum(axis=1)
        grad = 1/N * (exp.T/sum).T
    
        target_exp = input[:,target[:]].diag().exp()
       
        grad_target = -1/N * (sum-target_exp) / sum
        
        grad[range(len(grad)), target] = grad_target
        
        #for idx,idy in enumerate(target):
        #    grad[idx,idy.item()] = grad_target[idx]
            
        assert(grad.size() == input.size())
        return grad