from .operations import Accumulated, AddBackward, SubBackward, MultiBackward
from tensor.core import Tensor

ADD = "+"
SUB = "-"
MUL = "*"

def build_node_backward(nodeA, nodeB, operation):
    operation_next_functions = []
    
    if isinstance(nodeA, Tensor):
        if nodeA.is_leaf:
            operation_next_functions.push(Accumulated(nodeA))
    
    if isinstance(nodeB, Tensor):
        if nodeB.is_leaf:
            operation_next_functions.push(Accumulated(nodeB))
        
       