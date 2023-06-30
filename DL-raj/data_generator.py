
import numpy as np

np.random.seed(42)

def f(x,y):
    eq='x+y'
    x=str(x)
    y=str(y)
    eq=eq.replace('x',str(x),1)
    eq=eq.replace('y',str(y),1)
    return eval(eq)

def data_generator(m):
    x_list = [np.random.randn() for i in range(m)]
    y_list = [np.random.randn() for i in range(m)]

    data = []
    output = []
    for x,y in zip(x_list,y_list):
        data+=[[[x,y]]]
        output+=[[[f(x,y)]]]

    return np.array(data), np.array(output)

print (data_generator(3))

