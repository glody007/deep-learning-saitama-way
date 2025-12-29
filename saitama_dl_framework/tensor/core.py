from tensor.utils import (
    is_float,
    is_int,
    is_list,
    is_number
)


class Tensor:

    def __init__(self, data):
        self.data = data
        self.shape_array = []
        #self.__is_valid()
        
    def shape(self):
        self.__get_shape(self.data)
        return self.shape_array
            
            
    def mul(self, value):
        pass
        
        
    def add(self, value):
        pass
    
    def sub(self, value):
        pass
    
    def __get_shape(self, data):
        if not is_list(data):
            return
        if is_list(data):
            self.shape_array.append(len(data))
            self.__get_shape(data[0])
        
    def __is_valid(self):
        raise ValueError("")     
            