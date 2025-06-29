print("=== STARTING TEST ===")
print("Starting test_autograd_like.py")
from degrad.gradient import grad
import degrad.numpy_wrapper as anp
# import degrad.differentials  # Import to register differentials
import numpy as np


def f(x):
    print(f"f called with x={x}")
    # result = anp.add(anp.multiply(x, x), anp.add(anp.multiply(2, x), 1))
    exp_log = anp.exp(x) + anp.log(x)
    result = exp_log ** 2
    # print(f"f returns {result}")
    return result



grad_f = grad(f)
print("Created grad_f")

x = 3.0
computed_grad = grad_f(x)
print(f"computed_grad = {computed_grad}")
expected_grad = (f(x + 0.001) - f(x))/0.001

print(f"Computed grad: {computed_grad}")
print(f"Expected grad: {expected_grad}")
print(f"Test passed: {abs(computed_grad - expected_grad) < 1e-6}")
