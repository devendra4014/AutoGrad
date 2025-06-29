print("Importing degrad.numpy_wrapper")
import numpy as _np

from degrad.primitive import primitive

# List of NumPy functions we want to wrap
primitive_functions = [
    _np.log, _np.sin, _np.cos, _np.tan, _np.mod, _np.negative,
    _np.reciprocal, _np.exp, _np.sinh, _np.cosh, _np.tanh,
    _np.sqrt, _np.square, _np.add, _np.multiply, _np.subtract,
    _np.divide, _np.maximum, _np.minimum, _np.fmax, _np.fmin,
    _np.logaddexp, _np.logaddexp2, _np.mod,
    _np.remainder, _np.power, _np.arctan2, _np.hypot
]


# Function to wrap and store functions in the global namespace
def wrap_namespace(old_funcs, new_namespace):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int8, _np.int16, _np.int32, _np.int64, _np.integer}

    for obj in old_funcs:
        name = obj.__name__
        if callable(obj):
            new_namespace[name] = primitive(obj)  # Wrap and store function globally
        elif type(obj) in unchanged_types:
            new_namespace[name] = obj


# Apply wrapping to globals
wrap_namespace(primitive_functions, globals())
