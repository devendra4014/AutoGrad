from degrad.nodes import Node




def grad(fun):
    """
    Returns a function that computes the gradient of fun at a given input x.
    """

    def grad_fn(x):
        # Build computation graph
        root = Node.new_root(x)
        out = fun(root)
        # Zero gradients before backward
        out.zero_grad()
        # Backward pass
        out.backward(1.0)
        return root.grad

    return grad_fn
