"""
Mathematical utility functions.
"""
from typing import List


def MCD(n1: int, n2: int, lower_bound: int, upper_bound: int) -> List[int]:
    """
    Find common divisors between two numbers within a given range.
    
    Args:
        n1: First number
        n2: Second number
        lower_bound: Lower bound for divisor search
        upper_bound: Upper bound for divisor search
        
    Returns:
        List of common divisors
    """
    common_divisors = []
    for i in range(lower_bound + 1, upper_bound):
        if n1 % i == 0 and n2 % i == 0:
            common_divisors.append(i)
    return common_divisors 