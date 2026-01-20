from .backward import Accumulated, AddBackward, SubBackward, MultiBackward
from tensor.core import Tensor
from autograd import operations

def build_node_backward(nodeA, nodeB, operation):
    next_functions = []
    
    if isinstance(nodeA, Tensor):
        if nodeA.is_leaf:
            next_functions.push((Accumulated(nodeA), 0))
            
        elif nodeA.grad_fn:
            next_functions.push((nodeA.grad_fn, 0))
    
    if isinstance(nodeB, Tensor):
        if nodeB.is_leaf:
            next_functions.push(Accumulated(nodeB))
        
        elif nodeB.grad_fn:
            next_functions.push((nodeB.grad_fn, 0))
            
    
    if operation == operations.ADD:
        return AddBackward(next_functions)
    
    if operation == operations.SUB:
        return SubBackward(next_functions)
    
    if operation == operations.MUL:
        return MultiBackward(next_functions)
    
    raise ValueError("Invalid operation")