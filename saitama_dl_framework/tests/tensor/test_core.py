from tensor.core import Tensor    
    
def test_shape():
    scalar_shape = Tensor(1).shape()
    assert type(scalar_shape) == list
    assert len(scalar_shape) == 0
    
    vector_shape = Tensor([1, -1]).shape()
    assert len(vector_shape) == 1
    assert vector_shape[0] == 2
    
    tensor_shape_2_2 = Tensor([
        [2, 3], 
        [-1, 4]
    ]).shape()
    assert len(tensor_shape_2_2) == 2
    assert tensor_shape_2_2[0] == 2
    assert tensor_shape_2_2[1] == 2
    
    
    tensor_shape_3_2_2 = Tensor([
        [[2, 3], [-1, 4]], 
        [[2, 3], [-1, 4]], 
        [[2, 3], [-1, 4]]
    ]).shape()
    
    assert len(tensor_shape_3_2_2) == 3
    assert tensor_shape_3_2_2[0] == 3
    assert tensor_shape_3_2_2[1] == 2
    assert tensor_shape_3_2_2[2] == 2
    
    
    
    
