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
    assert result_vec.grad_fn.next_functions[0][0].node == v1
    assert result_vec.grad_fn.next_functions[1][0].node == v2
    
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
    assert result_vec.grad_fn.next_functions[0][0].node == v1
    assert result_vec.grad_fn.next_functions[1][0].node == v2
    
    
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
    assert result_vec.grad_fn.next_functions[0][0].node == v1
    assert result_vec.grad_fn.next_functions[1][0].node == v2
    
    
def test__graph_construction__():
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])
    v3 = Tensor([4, 3, 2, 1])
    v4 = Tensor([4, 3, 2, 1])
    v5 = Tensor([0, 3, 2, 1])
    v6 = Tensor([2, 30, 2, 5])

    result_mul1 = v1.multiply(v2)
    result_add = result_mul1 + v3
    result_sub = result_add - v4
    result_mul2 = v5.multiply(v6)
    result_merge_branch = result_mul1 + result_mul2

    assert result_mul1.is_leaf == False
    assert result_add.is_leaf == False
    assert result_sub.is_leaf == False
    
    grad_fn_sub = result_sub.grad_fn
    grad_fn_add = result_add.grad_fn
    grad_fn_mul1 = result_mul1.grad_fn
    grad_merge_branch = result_merge_branch.grad_fn
    
    
    assert isinstance(grad_fn_sub, backward.SubBackward)
    assert isinstance(grad_fn_sub.next_functions[0][0], backward.AddBackward)
    assert isinstance(grad_fn_sub.next_functions[1][0], backward.Accumulated)
    assert grad_fn_sub.next_functions[0][0] == result_add.grad_fn
    
    assert isinstance(grad_fn_add, backward.AddBackward)
    assert isinstance(grad_fn_add.next_functions[0][0], backward.MultiBackward)
    assert isinstance(grad_fn_add.next_functions[1][0], backward.Accumulated)
    assert grad_fn_add.next_functions[0][0] == result_mul1.grad_fn
    
    assert isinstance(grad_fn_mul1, backward.MultiBackward)
    assert isinstance(grad_fn_mul1.next_functions[0][0], backward.Accumulated)
    assert isinstance(grad_fn_mul1.next_functions[1][0], backward.Accumulated)
    assert grad_fn_mul1.next_functions[0][0].node == v1
    assert grad_fn_mul1.next_functions[1][0].node == v2
    
    assert isinstance(grad_merge_branch, backward.AddBackward)
    assert isinstance(grad_merge_branch.next_functions[0][0], backward.MultiBackward)
    assert isinstance(grad_merge_branch.next_functions[1][0], backward.MultiBackward)
    
    
    