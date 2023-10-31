""" General purpose utility functions
"""
import numpy as np
from functools import wraps
import time

TIMEIT_PRINT_ARGS = False   # Flag to pring out all args of the function being timed

def next_power_of_2(k):
    """Find the closest power of 2 larger than k.
    This is a nice solution suggested here:
    https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python

    Args:
        k (int > 0): input number

    Returns:
        n (int): minimal 2^m that is larger or equal than k

    Note: the function still does not crash for non-positive inputs but 
        results are wrong.
    """
    return 1<<(k-1).bit_length() 

def closest_elements(x, y):
    """For two sorted lists `x`, 'y', find elements in `x'
    closest to elements in `y`.

    Args:
        x (list of floats): sorted list of values to sample from in
            ascending order 
        y (list of floats): sorted list of values to be approximated 
            by values in `x` in ascending order

    Returns:
        idx (list of int): list of indecies of closest elements in x;
            `len(idx) = len(y)`
    """
    idx = list()    # List of indecies into x[]
    k0 = 0          # Starting index in x[] to search from
    nx = len(x)

    for i, yi in enumerate(y):
        for k in range(k0, nx):
            if x[k] > yi:
                break

        if np.abs(x[k] - yi) > np.abs(x[k-1] -yi):
            k0 = k - 1
        else:
            k0 = k

        idx.append(k0)

    return idx


def timeit(func):
    '''
    A decorator to use for benchmarking functions or class methods.
    Usage: add @timeit for your function/method

    Args:
        func (callable): functin to be timed

    Returns:
        wrapped (callable): the decorated function
    '''

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        if TIMEIT_PRINT_ARGS:
            print(f'Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds')
        else:
            print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

