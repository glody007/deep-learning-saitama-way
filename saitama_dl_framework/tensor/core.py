import copy
from tensor.utils import is_list


class Tensor:

    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.shape_list = self._get_shape(self.data)
        
    def shape(self):
        return self.shape_list
    
    def to_data(self):
        return copy.deepcopy(self.data)
            
            
    def multiply(self, tensor_2):
        if not is_list(self.data):
            return self.data * tensor_2.to_data()
        
    def matmul(self, tensor_2):   
        result =  [] 
        matrix_1_data = self.to_data()
        matrix_2_data = tensor_2.to_data()
        
        rows_count_matrix_1 = len(matrix_1_data) if self._is_matrix() else 1
        cols_count_matrix_1 = len(matrix_1_data[0]) if self._is_matrix() else len(matrix_1_data)
        
        # transform into a list of two dimension to be able to generalize multiplication to vector
        if self._is_vector():
            matrix_1_data = [matrix_1_data]
            
            
        result = []
        for i in range(rows_count_matrix_1):
            sum = 0
            for j in range(cols_count_matrix_1):
                sum += matrix_1_data[i][j] * matrix_2_data[j][i]
                print(f"i => {i} j=> {j} matrix => {matrix_1_data[i][j]} matrix => {matrix_2_data[j][i]} sum => {sum}")
            result.append(sum)

        return Tensor(result)

        
    def add(self, value):
        pass
    
    def sub(self, value):
        pass
    
    def _get_shape(self, data):
        shape = []
        
        if not is_list(data):
            return []
        
        if is_list(data) and data:
            dim = len(data)
            shape.append(dim)
            shape.extend(self._get_shape(data[0]))
            return shape
            
        return [] 
    
    def _is_scalar(self):
        if self.shape_list:
            return False
        return True
    
    def _is_vector(self):
        if len(self.shape_list) == 1:
            return True
        return False
    
    def _is_matrix(self):
        if len(self.shape_list) == 2:
            return True
        return False
            