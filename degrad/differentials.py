import numpy as np
from degrad import numpy_wrapper as anp

primitive_diff_func = {}


def register_diff(fun, *diff_lambda):
    print(f"Registering Function : {fun.__name__}")
    lambda_dict = {i: d_lambda for i, d_lambda in enumerate(diff_lambda)}

    def executable(argnums, ans, args, kwargs):
        if len(argnums) == 1:
            argnum = argnums[0]
            if argnum not in lambda_dict:
                raise NotImplementedError(f"VJP for {fun.__name__} wrt arg {argnum} not defined")
            vjp = lambda_dict[argnum](ans, *args, **kwargs)
            return lambda g: (vjp(g),)
        elif len(argnums) == 2:
            f0, f1 = lambda_dict[argnums[0]], lambda_dict[argnums[1]]
            vjp0 = f0(ans, *args, **kwargs)
            vjp1 = f1(ans, *args, **kwargs)
            return lambda g: (vjp0(g), vjp1(g))
        else:
            vjps = [lambda_dict[i](ans, *args, **kwargs) for i in argnums]
            return lambda g: tuple(vjp(g) for vjp in vjps)

    primitive_diff_func[fun] = executable


def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))


def replace_zero(x, val):
    return np.where(x, x, val)


def unbroadcast_f(target, f):
    target_meta = np.shape(target), np.ndim(target), np.result_type(target), np.iscomplexobj(target)
    return lambda g: f(g)


# ------ Single input functions ----------
register_diff(anp.log, lambda ans, x: lambda g: g / x)
register_diff(anp.sin, lambda ans, x: lambda g: g * anp.cos(x))
register_diff(anp.cos, lambda ans, x: lambda g: -g * anp.sin(x))
register_diff(anp.tan, lambda ans, x: lambda g: g / anp.cos(x) ** 2)
register_diff(anp.square, lambda ans, x: lambda g: g * 2 * x)
register_diff(anp.sqrt, lambda ans, x: lambda g: g * 0.5 * x**-0.5)
register_diff(anp.exp, lambda ans, x: lambda g: ans * g)
# All register_diff calls have been moved to numpy_wrapper.py

# ----- Binary Input Funtions -----

register_diff(
    anp.add, lambda ans, x, y: unbroadcast_f(x, lambda g: g), lambda ans, x, y: unbroadcast_f(y, lambda g: g)
)
register_diff(
    anp.multiply,
    lambda ans, x, y: unbroadcast_f(x, lambda g: y * g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: x * g),
)
register_diff(
    anp.subtract,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g),
)
register_diff(
    anp.divide,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g / y),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g * x / y ** 2),
)
register_diff(
    anp.maximum,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
register_diff(
    anp.minimum,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
register_diff(
    anp.fmax,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
register_diff(
    anp.fmin,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)),
)
register_diff(
    anp.logaddexp,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * np.exp(x - ans)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * np.exp(y - ans)),
)
register_diff(
    anp.logaddexp2,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * 2 ** (x - ans)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * 2 ** (y - ans)),
)
# register_diff(
#     anp.true_divide,
#     lambda ans, x, y: unbroadcast_f(x, lambda g: g / y),
#     lambda ans, x, y: unbroadcast_f(y, lambda g: -g * x / y ** 2),
# )
# register_diff(
#     anp.mod,
#     lambda ans, x, y: unbroadcast_f(x, lambda g: g),
#     lambda ans, x, y: unbroadcast_f(y, lambda g: -g * np.floor(x / y)),
# )
register_diff(
    anp.remainder,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g),
    lambda ans, x, y: unbroadcast_f(y, lambda g: -g * np.floor(x / y)),
)
register_diff(
    anp.power,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * y * x ** np.where(y, y - 1, 1.0)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * np.log(replace_zero(x, 1.0)) * ans),
)
register_diff(
    anp.arctan2,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * y / (x ** 2 + y ** 2)),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * -x / (x ** 2 + y ** 2)),
)
register_diff(
    anp.hypot,
    lambda ans, x, y: unbroadcast_f(x, lambda g: g * x / ans),
    lambda ans, x, y: unbroadcast_f(y, lambda g: g * y / ans),
)
