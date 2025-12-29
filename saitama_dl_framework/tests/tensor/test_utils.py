from tensor.utils import (
    is_int, 
    is_float, 
    is_list,
    is_number
)

def test_is_int():
    assert is_int(2) == True
    assert is_int([2]) == False
    assert is_int(4.2) == False
    assert is_int("saitama") == False
    
    
def test_is_float():
    assert is_float(2.0) == True
    assert is_float([2]) == False
    assert is_float(2) == False
    assert is_float("saitama") == False
    
def test_is_float():
    assert is_float(2.0) == True
    assert is_float([2]) == False
    assert is_float(2) == False
    assert is_float("saitama") == False
    
def test_is_number():
    assert is_number(2.0) == True
    assert is_number(2) == True
    assert is_number([4, 6]) == False
    assert is_number("saitama") == False
    
    
def test_is_list():
    assert is_list([2]) == True
    assert is_list([2, 4]) == True
    assert is_list(2.0) == False
    assert is_list(2) == False
    assert is_list("saitama") == False
    
    
    
    
    
