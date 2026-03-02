import saitama_dl as sdl
from saitama_dl.tensor.core import Tensor
from saitama_dl.autograd import backward

def test__no_grad():
    # starts enabled
    assert sdl._tracking_grad_enabled == True

    # disables inside with block
    with sdl.no_grad():
        assert sdl._tracking_grad_enabled == False

    # restores after exiting
    assert sdl._tracking_grad_enabled == True

    # nested: restores correctly
    with sdl.no_grad():
        with sdl.no_grad():
            assert sdl._tracking_grad_enabled == False
        assert sdl._tracking_grad_enabled == False
    assert sdl._tracking_grad_enabled == True

    # restores even if error happens
    try:
        with sdl.no_grad():
            raise ValueError("Error")
    except ValueError:
        pass
    assert sdl._tracking_grad_enabled == True
    
    
    from saitama_dl.autograd import backward

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

    with sdl.no_grad():
        result_vec = v1 - v2
        grad_fn = result_vec.grad_fn
        assert result_vec.is_leaf == False
        assert grad_fn == None
    
    
def test___mul__():
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])

    with sdl.no_grad():
        result_vec = v1.multiply(v2)
        grad_fn = result_vec.grad_fn
        assert result_vec.is_leaf == False
        assert grad_fn == None
    
    
def test__graph_construction__():
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])
    v3 = Tensor([4, 3, 2, 1])
    v4 = Tensor([4, 3, 2, 1])
    v5 = Tensor([0, 3, 2, 1])
    v6 = Tensor([2, 30, 2, 5])

    result_mul1 = v1.multiply(v2)
    
    with sdl.no_grad():
        result_add = result_mul1 + v3
        result_sub = result_add - v4
        result_mul2 = v5.multiply(v6)
        result_merge_branch = result_mul1 + result_mul2
    
        assert result_sub.grad_fn == None
        assert result_add.grad_fn == None 
        assert result_mul1.grad_fn != None
        assert result_merge_branch.grad_fn == None
    
    