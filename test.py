import numpy as np

import light


def main():
    # in_example = np.array([1, 1, 1, 1, 1])
    #
    # linear = light.Linear(5, 3, init_rand=True)
    #
    # print(linear.forward(in_example))
    # linear.backward(np.array([1, 2, 3]))
    # print(linear.grads['weight'].shape, linear.weight.shape)
    # print(id(linear.grads['weight']))
    #
    # sigmoid = light.Softmax()
    # print(sigmoid.forward(in_example))
    # print(sigmoid.backward(np.array([1, 2, 3, 4, 5])))

    class MyClass:
        def __init__(self):
            self.a = 10
            self.b = 20

        def my_method(self):
            return "Hello"

    obj = MyClass()

    for attr in dir(obj):
        print(attr, getattr(obj, attr), type(getattr(obj, attr)))

    my_dict = {}

    if not my_dict:
        print("The dictionary is empty")

if __name__ == '__main__':
    main()