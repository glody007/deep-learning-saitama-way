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
    tensor_2 = Tensor([
        [1], [2]
    ])
    tensor_3 = Tensor([
        [1, 6], 
        [3, 2]
    ])
    tensor_4 = Tensor([
        [-3, 2], 
        [1, 2]
    ])
    tensor_5 = Tensor([
        [-3, 2, 5], 
        [10, 1, 2]
    ])
    tensor_6 = Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    tensor_7 = Tensor([
        [7, 8],
        [9, 10],
        [11, 12]
    ])
    
    
    tensor_result_1 = tensor_1.matmul(tensor_2)
    assert tensor_result_1.to_data() == [5]
        
    tensor_result_2 = tensor_1.matmul(tensor_3)
    assert tensor_result_2.to_data() == [7, 10]
    
    tensor_result_3 = tensor_3.matmul(tensor_4)
    assert tensor_result_3.to_data() == [[3, 14], [-7, 10]]
    
    tensor_result_4 = tensor_4.matmul(tensor_5)
    assert tensor_result_4.to_data() == [[29, -4, -11], [17, 4, 9]]
    
    tensor_result_5 = tensor_6.matmul(tensor_7)
    assert tensor_result_5.to_data() == [[58, 64], [139, 154]]
    
    
def test_add():
    # vector addition
    v1 = Tensor([1, 2, 3, 4])
    v2 = Tensor([4, 3, 2, 1])

    result_vec = v1.add(v2)
    assert result_vec.to_data() == [5, 5, 5, 5]

    # 4x4 matrix addition
    m1 = Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    m2 = Tensor([
        [16, 15, 14, 13],
        [12, 11, 10, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1]
    ])

    result_mat = m1.add(m2)
    assert result_mat.to_data() == [
        [17, 17, 17, 17],
        [17, 17, 17, 17],
        [17, 17, 17, 17],
        [17, 17, 17, 17]
    ]


def test_subs():
    # vector subtraction
    v1 = Tensor([10, 20, 30, 40])
    v2 = Tensor([1, 2, 3, 4])

    result_vec = v1.sub(v2)
    assert result_vec.to_data() == [9, 18, 27, 36]

    # 4x4 matrix subtraction
    m1 = Tensor([
        [20, 20, 20, 20],
        [20, 20, 20, 20],
        [20, 20, 20, 20],
        [20, 20, 20, 20]
    ])

    m2 = Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    result_mat = m1.sub(m2)
    assert result_mat.to_data() == [
        [19, 18, 17, 16],
        [15, 14, 13, 12],
        [11, 10, 9, 8],
        [7, 6, 5, 4]
    ]


def test_element_wise_multiplication():
    # vector element-wise multiplication
    v1 = Tensor([2, 3, 4, 5])
    v2 = Tensor([5, 4, 3, 2])

    result_vec = v1.multiply(v2)
    assert result_vec.to_data() == [10, 12, 12, 10]

    # 4x4 matrix element-wise multiplication
    m1 = Tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7]
    ])

    m2 = Tensor([
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5]
    ])

    result_mat = m1.multiply(m2)
    assert result_mat.to_data() == [
        [2, 4, 6, 8],
        [6, 9, 12, 15],
        [12, 16, 20, 24],
        [20, 25, 30, 35]
    ]





    

    

    
    
    
    
    
    
    
