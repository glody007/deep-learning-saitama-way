from tensor.utils import (
    is_float,
    is_int,
    is_list,
    is_number
)


class Tensor:

    def __init__(self, data):
        self.data = data
        self.shape_list = self.__get_shape(self.data)
        
    def shape(self):
        return self.shape_list
            
            
    def mul(self, value):
        pass
        
        
    def add(self, value):
        pass
    
    def sub(self, value):
        pass
    
    def __get_shape(self, data):
        shape = []
        
        if not is_list(data):
            return []
        
        if is_list(data) and data:
            dim = len(data)
            shape.append(dim)
            shape.extend(self.__get_shape(data[0]))
            return shape
            
        return []
        
    def __is_valid(self):
        raise ValueError("")     
            