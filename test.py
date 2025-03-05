import numpy as np

import light


def main():
    in_example = np.array([1, 1, 1, 1, 1])

    linear = light.Linear(5, 3, init_rand=True)

    print(linear.forward(in_example))
    linear.backward(np.array([1, 2, 3]))
    print(linear.grads['weight'].shape, linear.weight.shape)

    sigmoid = light.Softmax()
    print(sigmoid.forward(in_example))
    print(sigmoid.backward(np.array([1, 2, 3, 4, 5])))

if __name__ == '__main__':
    main()