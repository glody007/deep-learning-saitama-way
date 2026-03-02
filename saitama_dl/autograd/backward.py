
class Accumulated():
    node = None
    
    def __init__(self, node):
        self.node = node
    
    def backward(self, grad):
        if self.node.grad == None:
            self.node.set_grad(grad)
        else: 
            self.node.set_grad(self.node.grad + grad)

class MultiBackward():
    next_functions = []
    saved_tensors = []
    
    def __init__(self, next_functions, saved_tensors):
        self.next_functions = next_functions
        self.saved_tensors = saved_tensors
        
    def backward(self, parent_grad):
        self.next_functions[0][0].backward(parent_grad*self.saved_tensors[1])
        self.next_functions[1][0].backward(parent_grad*self.saved_tensors[0])


class AddBackward():
    next_functions = []
    saved_tensors = []
    
    def __init__(self, next_functions, saved_tensors):
        self.next_functions = next_functions
        self.saved_tensors = saved_tensors
        
    def backward(self, parent_grad):
        for next_function in self.next_functions:
            next_function[0].backward(parent_grad)
        
class SubBackward():
    next_functions = []
    saved_tensors = []
    
    def __init__(self, next_functions, saved_tensors):
        self.next_functions = next_functions
        self.saved_tensors = saved_tensors
        
    def backward(self, parent_grad):
        for next_function in self.next_functions:
            next_function[0].backward(parent_grad)