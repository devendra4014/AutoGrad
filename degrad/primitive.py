import functools


def primitive(f_raw):
    @functools.wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        # Import Node here to avoid circular import
        from degrad.nodes import Node
        
        # If any argument is a Node, build the computation graph
        if any(isinstance(arg, Node) for arg in args):
            # Track which arguments were Node objects and their positions
            node_indices = [i for i, arg in enumerate(args) if isinstance(arg, Node)]
            parents = tuple([arg for arg in args if isinstance(arg, Node)])
            
            # Get the values for computation (Node values for Node objects, original values for others)
            argvals = tuple([arg._value if isinstance(arg, Node) else arg for arg in args])
            
            ans = f_raw(*argvals, **kwargs)
            node = Node(ans, f_wrapped, parents, node_indices, original_args=args)
            return node
        else:
            return f_raw(*args, **kwargs)
    return f_wrapped


def _func(func):
    @functools.wraps(func)
    def apply_func(function, args):
        result = function(func, args)
        return result._value

    return apply_func

@_func
def grad(fun, x):
    # Import Node here to avoid circular import
    from degrad.nodes import Node
    new_node = Node.new_root(x)
    return fun(new_node)