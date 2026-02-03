from tensor.core import Tensor  
from autograd import backward

def test___add__():
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])

    result_vec = v1 + v2
    grad_fn = result_vec.grad_fn
    assert result_vec.is_leaf == False
    assert grad_fn != None
    assert isinstance(grad_fn, backward.AddBackward)
    assert isinstance(grad_fn.next_functions[0][0], backward.Accumulated)
    assert isinstance(grad_fn.next_functions[1][0], backward.Accumulated)
    
def test___sub__():
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])

    result_vec = v1 - v2
    grad_fn = result_vec.grad_fn
    assert result_vec.is_leaf == False
    assert grad_fn != None
    assert isinstance(grad_fn, backward.SubBackward)
    assert isinstance(grad_fn.next_functions[0][0], backward.Accumulated)
    assert isinstance(grad_fn.next_functions[1][0], backward.Accumulated)
    
    
def test___mul__():
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])

    result_vec = v1.multiply(v2)
    grad_fn = result_vec.grad_fn
    assert result_vec.is_leaf == False
    assert grad_fn != None
    assert isinstance(grad_fn, backward.MultiBackward)
    assert isinstance(grad_fn.next_functions[0][0], backward.Accumulated)
    assert isinstance(grad_fn.next_functions[1][0], backward.Accumulated)