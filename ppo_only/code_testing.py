import numpy as np
import time
import math
import matplotlib.pyplot as plt

numpy_gen = np.random.default_rng(0)

vals = []

def test(lower_bound, upper_bound, mode, rpy_flag=0):
    start = np.array([[0.0, 0.0, 0.0]])
    save = start.copy()
    if(mode == 0):
        sample = numpy_gen.uniform(low=lower_bound, high=upper_bound,
                size=1)
        start[0][rpy_flag] += sample[0]
    else:
        sample = numpy_gen.uniform(low=lower_bound, high=upper_bound, size=1)
        sample_var = numpy_gen.random()
        val = math.floor(sample_var+0.5)
        adder = (-1)*(1-val)*(sample[0]) + (val)*(sample[0])
        start[0][rpy_flag] += adder
    vals.append(start[0][rpy_flag])
    start = save.copy()

def runner():
    mode = int(input('enter mode to test\n'))
    if(mode%2):
        low = 0
        upper=0
        prev=0
        for i in range(19, -1, -1):
            for j in range(100):
                if(mode == 0):
                    low = -(i*(3.14/20))
                    upper = i*(3.14/20)
                    test(low, upper, mode, 0)
                elif(mode == 1):
                    low = i*(3.14/20)
                    upper = 3.14
                    test(low, upper, mode, 0)
                elif(mode == 2):
                    low = ((i-1)*(3.14/20))
                    upper = i*(3.14/20)
                    test(low, upper, mode, 0)
                elif(mode == 3):
                    low =(i*(3.14/20))
                    upper = (i+1)*(3.14/20)
                    test(low, upper, mode, 0)
            xs = [0]*(len(vals))
            title = f'low={low} upper={upper} prev={prev}\
                mode = {mode}'
            fig, ax = plt.subplots()
            fig = plt.scatter(vals, xs)
            plt.xlim(-3.14, 3.14)
            plt.ylim(-0.1, 0.1)
            ax.set_title(title)
            plt.show()
            vals.clear()
    else:
        low = 0
        upper=0
        prev=0
        for i in range(1, 19):
            for j in range(100):
                if(mode == 0):
                    low = -(i*(3.14/20))
                    upper = i*(3.14/20)
                elif(mode == 1):
                    low = i*(3.14/20)
                    upper = 3.14
                elif(mode == 2):
                    low = ((i-1)*(3.14/20))
                    upper = i*(3.14/20)
                elif(mode == 3):
                    low =(i*(3.14/20))
                    upper = (i+1)*(3.14/20)
                test(low, upper, mode, 0)
            xs = [0]*(len(vals))
            title = f'low={low} upper={upper} prev={prev}\
                mode = {mode}'
            fig, ax = plt.subplots()
            fig = plt.scatter(vals, xs)
            plt.xlim(-3.14, 3.14)
            plt.ylim(-0.1, 0.1)
            ax.set_title(title)
            plt.show()
            vals.clear()

def runner1():
    a=int(input("Enter 0 or 1"))
    b = (a == 1)*10
    print(b)

runner1()
