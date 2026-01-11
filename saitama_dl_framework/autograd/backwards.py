
class Accumulated():
    node = None
    
    def __init__(self, node):
        self.node = node
    
    def apply(self, grad):
        self.node.set_grad(grad)


class MultiBackward():
    next_functions = []


class AddBackward():
    next_functions = []