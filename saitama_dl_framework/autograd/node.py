

class Node():
    
    data = None
    grad = None
    grad_fn = None
    is_leaf = False
    
    def __init__(self):
        pass
    
    def set_grad(self, grad):
        self.grad = grad