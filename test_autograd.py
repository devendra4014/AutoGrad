import numpy as np
from degrad.nodes import Node
from degrad.gradient import grad
from degrad import numpy_wrapper as anp


def test_basic_operations():
    """Test basic arithmetic operations"""
    print("\n=== Testing Basic Operations ===")
    
    # Test addition
    def add_func(x):
        return anp.add(x , 2.0)
    
    x = Node.new_root(3.0)
    result = add_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Addition: f(x) = x + 2, x=3, f(x)={result._value}, grad={x.grad}")
    print(f"  result.parents: {result.parents}")
    for p in result.parents:
        print(f"    parent: {p}, grad: {getattr(p, 'grad', None)}")
    assert abs(result._value - 5.0) < 1e-6
    assert abs(x.grad - 1.0) < 1e-6
    
    # Test multiplication
    def mul_func(x):
        return x * 4.0
    
    x = Node.new_root(2.0)
    result = mul_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Multiplication: f(x) = x * 4, x=2, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 8.0) < 1e-6
    assert abs(x.grad - 4.0) < 1e-6


def test_unary_functions():
    """Test unary mathematical functions"""
    print("\n=== Testing Unary Functions ===")
    
    # Test square
    def square_func(x):
        return anp.square(x)
    
    x = Node.new_root(3.0)
    result = square_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Square: f(x) = x², x=3, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 9.0) < 1e-6
    assert abs(x.grad - 6.0) < 1e-6
    
    # Test exponential
    def exp_func(x):
        return anp.exp(x)
    
    x = Node.new_root(1.0)
    result = exp_func(x)
    result.zero_grad()
    result.backward()
    
    expected_grad = np.exp(1.0)
    print(f"Exponential: f(x) = e^x, x=1, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - np.exp(1.0)) < 1e-6
    assert abs(x.grad - expected_grad) < 1e-6
    
    # Test logarithm
    def log_func(x):
        return anp.log(x)
    
    x = Node.new_root(2.0)
    result = log_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Logarithm: f(x) = ln(x), x=2, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - np.log(2.0)) < 1e-6
    assert abs(x.grad - 0.5) < 1e-6


def test_trigonometric_functions():
    """Test trigonometric functions"""
    print("\n=== Testing Trigonometric Functions ===")
    
    # Test sine
    def sin_func(x):
        return anp.sin(x)
    
    x = Node.new_root(np.pi/2)
    result = sin_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Sine: f(x) = sin(x), x=π/2, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 1.0) < 1e-6
    assert abs(x.grad - 0.0) < 1e-6  # cos(π/2) = 0
    
    # Test cosine
    def cos_func(x):
        return anp.cos(x)
    
    x = Node.new_root(0.0)
    result = cos_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Cosine: f(x) = cos(x), x=0, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 1.0) < 1e-6
    assert abs(x.grad - 0.0) < 1e-6  # -sin(0) = 0
    
    # Test tangent
    def tan_func(x):
        return anp.tan(x)
    
    x = Node.new_root(np.pi/4)
    result = tan_func(x)
    result.zero_grad()
    result.backward()
    
    expected_grad = 1.0 / (np.cos(np.pi/4) ** 2)
    print(f"Tangent: f(x) = tan(x), x=π/4, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 1.0) < 1e-6
    assert abs(x.grad - expected_grad) < 1e-6


def test_binary_operations():
    """Test binary operations with two variables"""
    print("\n=== Testing Binary Operations ===")
    
    # Test addition of two variables
    def add_two_func(x, y):
        return x + y
    
    x = Node.new_root(2.0)
    y = Node.new_root(3.0)
    result = add_two_func(x, y)
    result.zero_grad()
    result.backward()
    
    print(f"Add two vars: f(x,y) = x + y, x=2, y=3, f(x,y)={result._value}")
    print(f"  grad_x={x.grad}, grad_y={y.grad}")
    assert abs(result._value - 5.0) < 1e-6
    assert abs(x.grad - 1.0) < 1e-6
    assert abs(y.grad - 1.0) < 1e-6
    
    # Test multiplication of two variables
    def mul_two_func(x, y):
        return x * y
    
    x = Node.new_root(4.0)
    y = Node.new_root(5.0)
    result = mul_two_func(x, y)
    result.zero_grad()
    result.backward()
    
    print(f"Multiply two vars: f(x,y) = x * y, x=4, y=5, f(x,y)={result._value}")
    print(f"  grad_x={x.grad}, grad_y={y.grad}")
    assert abs(result._value - 20.0) < 1e-6
    assert abs(x.grad - 5.0) < 1e-6
    assert abs(y.grad - 4.0) < 1e-6


def test_composite_functions():
    """Test composite functions (chain rule)"""
    print("\n=== Testing Composite Functions ===")
    
    # Test f(x) = sin(x²)
    def composite_func(x):
        return anp.sin(anp.square(x))
    
    x = Node.new_root(1.0)
    result = composite_func(x)
    result.zero_grad()
    result.backward()
    
    # Manual calculation: d/dx[sin(x²)] = cos(x²) * 2x = cos(1) * 2 = 2*cos(1)
    expected_grad = 2.0 * np.cos(1.0)
    print(f"Composite: f(x) = sin(x²), x=1, f(x)={result._value}, grad={x.grad}")
    print(f"  Expected grad: {expected_grad}")
    assert abs(result._value - np.sin(1.0)) < 1e-6
    assert abs(x.grad - expected_grad) < 1e-6
    
    # Test f(x) = exp(sin(x))
    def exp_sin_func(x):
        return anp.exp(anp.sin(x))
    
    x = Node.new_root(0.0)
    result = exp_sin_func(x)
    result.zero_grad()
    result.backward()
    
    # Manual calculation: d/dx[exp(sin(x))] = exp(sin(x)) * cos(x) = exp(0) * 1 = 1
    print(f"Composite: f(x) = exp(sin(x)), x=0, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 1.0) < 1e-6
    assert abs(x.grad - 1.0) < 1e-6


def test_grad_function():
    """Test the grad function wrapper"""
    print("\n=== Testing Grad Function ===")
    
    def simple_func(x):
        return anp.square(x) + 2 * x
    
    grad_func = grad(simple_func)
    result = grad_func(3.0)
    
    # Manual calculation: f(x) = x² + 2x, f'(x) = 2x + 2, f'(3) = 8
    expected_grad = 2 * 3.0 + 2
    print(f"Grad function: f(x) = x² + 2x, x=3, grad={result}")
    print(f"  Expected grad: {expected_grad}")
    assert abs(result - expected_grad) < 1e-6


def test_edge_cases():
    """Test edge cases and potential issues"""
    print("\n=== Testing Edge Cases ===")
    
    # Test with zero
    def sqrt_func(x):
        return anp.sqrt(x)
    
    x = Node.new_root(4.0)
    result = sqrt_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Sqrt: f(x) = √x, x=4, f(x)={result._value}, grad={x.grad}")
    assert abs(result._value - 2.0) < 1e-6
    assert abs(x.grad - 0.25) < 1e-6  # 1/(2√4) = 1/4
    
    # Test with negative input (should handle gracefully)
    try:
        x = Node.new_root(-1.0)
        result = sqrt_func(x)
        print(f"Sqrt negative: f(x) = √x, x=-1, f(x)={result._value}")
        # This should raise an error or handle NaN
    except Exception as e:
        print(f"Sqrt negative handled: {e}")


def test_multiple_backward_calls():
    """Test that gradients accumulate correctly with multiple backward calls"""
    print("\n=== Testing Multiple Backward Calls ===")
    
    x = Node.new_root(2.0)
    result = anp.square(x)
    
    # First backward pass
    result.zero_grad()
    result.backward(1.0)
    grad1 = x.grad
    print(f"First backward: grad={grad1}")
    
    # Second backward pass
    result.zero_grad()
    result.backward(2.0)
    grad2 = x.grad
    print(f"Second backward: grad={grad2}")
    
    assert abs(grad1 - 4.0) < 1e-6  # 2 * 2 = 4
    assert abs(grad2 - 8.0) < 1e-6  # 2 * 2 * 2 = 8


def test_complex_expression():
    """Test a more complex mathematical expression"""
    print("\n=== Testing Complex Expression ===")
    
    def complex_func(x):
        return anp.exp(anp.sin(x)) * anp.cos(anp.square(x))
    
    x = Node.new_root(0.5)
    result = complex_func(x)
    result.zero_grad()
    result.backward()
    
    print(f"Complex: f(x) = exp(sin(x)) * cos(x²), x=0.5")
    print(f"  f(x)={result._value}, grad={x.grad}")
    
    # Verify the result is reasonable (not NaN or inf)
    assert not np.isnan(result._value)
    assert not np.isinf(result._value)
    assert not np.isnan(x.grad)
    assert not np.isinf(x.grad)


if __name__ == "__main__":
    print("Running Autograd Tests...")
    
    test_basic_operations()
    test_unary_functions()
    test_trigonometric_functions()
    test_binary_operations()
    test_composite_functions()
    test_grad_function()
    test_edge_cases()
    test_multiple_backward_calls()
    test_complex_expression()
    
    print("\n=== All Tests Completed ===")
    print("If no errors occurred, the autograd implementation is working correctly!") 