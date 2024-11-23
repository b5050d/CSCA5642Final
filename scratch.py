def normalize_cart_position(cpos):
    """
    Normalize the position of the cart to be between -1 and 1
    """
    return round(cpos/4.8,3)

def normalize_cart_speed(cspeed):
    """
    Normalize the speed of the cart to be betweeen -1 and 1
    """
    return round(cspeed/5, 3)

def normalize_pole_angle(pang):
    """
    Normalize the pole angle of the cart to be betweeen -1 and 1
    """
    return round(pang/.418, 3)

def normalize_pole_angular_velocity(pvel):
    """
    Normalize the pole angle of the cart to be betweeen -1 and 1
    """
    return round(pvel/5, 3)

import numpy as np
def normalize_data(state):
    """
    A Faster way to normalize the data
    """    
    normalization_factors = np.array([4.8, 5, .418, 5])
    return state/normalization_factors



a = np.array([[-0.03037071,-0.02083859,0.00267074,-0.0218708]])
print(a)
print(a.shape)
(1, 4)