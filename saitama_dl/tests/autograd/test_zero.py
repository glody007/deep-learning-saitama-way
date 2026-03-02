from saitama_dl.tensor.core import Tensor  

def test__zero_simple():
    v1 = Tensor([2])
    v2 = Tensor([1])
    
    simple_add = v1 + v2
    
    simple_add.backward()
    
    assert v1.grad.to_data() == [1]
    assert v2.grad.to_data() == [1]
    
    v1.grad.zero_()
    
    assert v1.grad.to_data() == [0]
    assert v2.grad.to_data() == [1]
    
def test__zero_on_multiple_tensors():
    v1 = Tensor([2])
    v2 = Tensor([1])
    v3 = Tensor([6])
    v4 = Tensor([4])

    result = (v1 + v2) * (v3 + v4)
    result.backward()

    # Zero only some grads
    v1.grad.zero_()
    v3.grad.zero_()

    assert v1.grad.to_data() == [0]
    assert v2.grad.to_data() == [10]  
    assert v3.grad.to_data() == [0]
    assert v4.grad.to_data() == [3]  

    # Zero the rest
    v2.grad.zero_()
    v4.grad.zero_()

    assert v2.grad.to_data() == [0]
    assert v4.grad.to_data() == [0]

    