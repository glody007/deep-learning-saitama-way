import copy
from tensor.utils import is_list
from autograd.graph import build_node_backward
from autograd import operations
class Tensor:
    
    grad = None
    grad_fn = None
    is_leaf = True
    required_grad = True

    def __init__(self, data, is_leaf=True, grad_fn=None, required_grad=True):
        self.data = copy.deepcopy(data)
        self.shape_list = self._get_shape(self.data)
        self.is_leaf = is_leaf
        self.grad_fn = grad_fn
        self.required_grad = required_grad
        
    def shape(self):
        return self.shape_list
    
    def to_data(self):
        return copy.deepcopy(self.data)
            
            
    def multiply(self, tensor_2):
        if not is_list(self.data):
            return self.data * tensor_2.to_data()
        
    def matmul(self, tensor_2):   
        matrix_A_data = self.to_data()
        matrix_B_data = tensor_2.to_data()
        
        rows_count_matrix_A = len(matrix_A_data) if self._is_matrix() else 1
        cols_count_matrix_A = len(matrix_A_data[0]) if self._is_matrix() else len(matrix_A_data)
        
        cols_count_matrix_B = len(matrix_B_data[0])
        
        # transform into a list of two dimension to be able to generalize multiplication to vector
        if self._is_vector():
            matrix_A_data = [matrix_A_data]
            
        matrix_C_data = []
        # for row Ai
        for A_row_i in range(rows_count_matrix_A):
            C_row = []
            # for column Bj
            for B_col_j in range(cols_count_matrix_B):
                # Ai*Bj: dot product = sum all Aik * Bjk
                sum = 0
                for A_col_j in range(cols_count_matrix_A):
                    sum += matrix_A_data[A_row_i][A_col_j] * matrix_B_data[A_col_j][B_col_j]
                C_row.append(sum)
            matrix_C_data.append(C_row)
            
        if self._is_vector():
            matrix_C_data = matrix_C_data[0]

        return Tensor(matrix_C_data)

        
    def add(self, tensor_2):   
        grad_fn = build_node_backward(self, tensor_2, operation=operations.ADD)
        return Tensor._element_wise_operation(
            self,
            tensor_2,
            lambda a, b: a + b,
            grad_fn=grad_fn
        )
    
    def sub(self, tensor_2):
        grad_fn = build_node_backward(self, tensor_2, operation=operations.SUB)
        return Tensor._element_wise_operation(
            self,
            tensor_2,
            lambda a, b: a - b,
            grad_fn=grad_fn
        )
        
    def multiply(self, tensor_2):
        grad_fn = build_node_backward(self, tensor_2, operation=operations.MUL)
        return Tensor._element_wise_operation(
            self,
            tensor_2,
            lambda a, b: a * b, 
            grad_fn=grad_fn
        )
        
    def __add__(self, tensor_2):
        return self.add(tensor_2)
    
    def __sub__(self, tensor_2):
        return self.sub(tensor_2)
    
    def __mul__(self, tensor_2):
        return self.multiply(tensor_2)
    
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
    
    # operation is function that takes two elements and apply the operation
    def _element_wise_operation(tensor_1, tensor_2, operation, grad_fn):
        matrix_A_data = tensor_1.to_data()
        matrix_B_data = tensor_2.to_data()
        
        rows_count_matrix_A = len(matrix_A_data) if tensor_1._is_matrix() else 1
        cols_count_matrix_A = len(matrix_A_data[0]) if tensor_1._is_matrix() else len(matrix_A_data)
        
        
        # transform into a list of two dimension to be able to generalize
        if tensor_1._is_vector():
            matrix_A_data = [matrix_A_data]
            
        if tensor_2._is_vector():
            matrix_B_data = [matrix_B_data]
            
        matrix_C_data = []
        # for row i
        for i in range(rows_count_matrix_A):
            C_row = []
            # for column j
            for j in range(cols_count_matrix_A):
                # operation(Aij, Bij)
                C_row.append(operation(matrix_A_data[i][j], matrix_B_data[i][j]))
            matrix_C_data.append(C_row)
            
        if tensor_1._is_vector():
            matrix_C_data = matrix_C_data[0]

        return Tensor(matrix_C_data, is_leaf=False, grad_fn=grad_fn)
    
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
            