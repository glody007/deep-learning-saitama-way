def is_number(value):
    return is_float(value) or is_int(value)
    
def is_int(value):
    return type(value) == int

def is_list(value):
    return type(value) == list      

def is_float(value):
    return type(value) == float   

def is_boolean(value):
    return type(value) == bool