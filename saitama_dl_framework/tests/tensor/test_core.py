from tensor.core import Tensor    
    
def test_shape():
    scalar_shape = Tensor(1).shape()
    assert scalar_shape == []
    
    vector_shape = Tensor([1, -1]).shape()
    assert vector_shape == [2]
    
    tensor_shape_2_2 = Tensor([
        [2, 3], 
        [-1, 4]
    ]).shape()
    assert tensor_shape_2_2 == [2, 2]
    
    
    tensor_shape_3_2_2 = Tensor([
        [[2, 3], [-1, 4]], 
        [[2, 3], [-1, 4]], 
        [[2, 3], [-1, 4]]
    ]).shape()
    assert tensor_shape_3_2_2 == [3, 2, 2]
    

    
def test_matmul():
    tensor_1 = Tensor([1, 2])
    tensor_2 = Tensor([[1], [2]])
    tensor_3 = Tensor([[1, 6], [3, 2]])
    tensor_4 = Tensor([[-3, 2], [1, 2]])
    tensor_5 = Tensor([[-3, 2, 5], [10, 1, 2]])
    
    
    tensor_result_1 = tensor_1.matmul(tensor_2)
    tensor_result_2 = tensor_1.matmul(tensor_3)
    tensor_result_3 = tensor_3.matmul(tensor_4)
    tensor_result_4 = tensor_4.matmul(tensor_5)
    
    assert tensor_result_1.to_data() == [5]

    assert tensor_result_2.to_data() == [7, 10]

    assert tensor_result_3.to_data() == [[3, 14], [-7, 10]]

    assert tensor_result_4.to_data() == [[29, -4, -11], [17, 3, 9]]

    
    
    
    
    
    
    
