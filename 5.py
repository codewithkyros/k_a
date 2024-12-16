import numpy as np

def mcculloch_pitts_neuron(inputs, weights, threshold):
    # Calculate the weighted sum of inputs
    weighted_sum = np.dot(inputs, weights)
    # Apply the threshold function
    return 1 if weighted_sum >= threshold else 0

def AND(x1, x2):
    weights = [1, 1]  # Both inputs are excitatory
    threshold = 2     # Both must be active to fire
    return mcculloch_pitts_neuron([x1, x2], weights, threshold)

def OR(x1, x2):
    weights = [1, 1]  # Both inputs are excitatory
    threshold = 1     # At least one must be active to fire
    return mcculloch_pitts_neuron([x1, x2], weights, threshold)

def XOR(x1, x2):
    # XOR is implemented using two neurons
    not_x2 = 1 - x2  # Inverse of x2
    first_neuron = AND(x1, not_x2)
    not_x1 = 1 - x1  # Inverse of x1
    second_neuron = AND(not_x1, x2)
    return OR(first_neuron, second_neuron)

def ANDNOT(x1, x2):
    weights = [1, -1]  # First input is excitatory; second is inhibitory
    threshold = 1      # Fires only if first input is active and second is inactive
    return mcculloch_pitts_neuron([x1, x2], weights, threshold)

def test_logic_functions():
    print("AND Function:")
    for i in range(2):
        for j in range(2):
            print(f"{i} AND {j} = {AND(i, j)}")

    print("\nOR Function:")
    for i in range(2):
        for j in range(2):
            print(f"{i} OR {j} = {OR(i, j)}")

    print("\nXOR Function:")
    for i in range(2):
        for j in range(2):
            print(f"{i} XOR {j} = {XOR(i, j)}")

    print("\nANDNOT Function:")
    for i in range(2):
        for j in range(2):
            print(f"{i} ANDNOT {j} = {ANDNOT(i, j)}")

if __name__ == "__main__":
    test_logic_functions()
