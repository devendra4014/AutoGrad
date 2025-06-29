import operator

from degrad.differentials import primitive_diff_func
import degrad.numpy_wrapper as anp

class Node:
    def __init__(self, value, func=None, parents=(), node_indices=None, original_args=None):
        self._value = value
        self.func = func
        self.parents = parents
        self.node_indices = node_indices
        self.original_args = original_args
        self.grad = 0.0
        self._vjpmaker = primitive_diff_func.get(func, None)

    def get_value(self):
        return self._value

    def initialize_root(self, val):
        self._value = val
        self.func = None
        self.parents = ()
        self.node_indices = None
        self.original_args = None
        self.grad = 0.0
        self._vjpmaker = None

    @classmethod
    def new_root(cls, val):
        return cls(val)

    def __neg__(self):
        return anp.negative(self)

    def __add__(self, other):
        return anp.add(self, other)

    def __sub__(self, other):
        return anp.subtract(self, other)

    def __mul__(self, other):
        return anp.multiply(self, other)

    def __pow__(self, other):
        return anp.power(self, other)

    def __div__(self, other):
        return anp.divide(self, other)

    def __mod__(self, other):
        return anp.mod(self, other)

    def __truediv__(self, other):
        return anp.true_divide(self, other)

    def __matmul__(self, other):
        return anp.matmul(self, other)

    def __radd__(self, other):
        return anp.add(other, self)

    def __rsub__(self, other):
        return anp.subtract(other, self)

    def __rmul__(self, other):
        return anp.multiply(other, self)

    def __rpow__(self, other):
        return anp.power(other, self)

    def __rdiv__(self, other):
        return anp.divide(other, self)

    def __rmod__(self, other):
        return anp.mod(other, self)

    def __rtruediv__(self, other):
        return anp.true_divide(other, self)

    def __rmatmul__(self, other):
        return anp.matmul(other, self)

    def __eq__(self, other):
        return anp.equal(self, other)

    def __ne__(self, other):
        return anp.not_equal(self, other)

    def __gt__(self, other):
        return anp.greater(self, other)

    def __ge__(self, other):
        return anp.greater_equal(self, other)

    def __lt__(self, other):
        return anp.less(self, other)

    def __le__(self, other):
        return anp.less_equal(self, other)

    def __abs__(self):
        return anp.abs(self)

    def __hash__(self):
        return id(self)

    def backward(self, grad_output=1.0):
        self.grad = grad_output
        topo_order = list(Node._toposort(self))
        print(f"Processing order (from output to input): {[f'Node({n._value:.2f})' for n in topo_order]}")
        for i, node in enumerate(topo_order):
            print(f"Step {i + 1}: Processing Node({node._value:.2f}) with grad={node.grad:.2f}")
            if node.func and node._vjpmaker and node.node_indices is not None:
                print(
                    f"  node.func: {getattr(node.func, '__name__', str(node.func))}, node_indices: {node.node_indices}")
                
                # Use the original arguments that were passed to the function
                if node.original_args is not None:
                    # Convert Node objects to their values for the differential computation
                    all_args = []
                    for arg in node.original_args:
                        if isinstance(arg, Node):
                            all_args.append(arg._value)
                        else:
                            all_args.append(arg)
                else:
                    # Fallback: just use parent values
                    all_args = [p._value for p in node.parents]
                
                print(f"    all_args: {all_args}")
                print(f"    vjp function: {node._vjpmaker}")
                vjp = node._vjpmaker(node.node_indices, node._value, tuple(all_args), {})
                grads = tuple(vjp(node.grad))
                print(f"    grads returned: {grads}")
                for parent, g in zip(node.parents, grads):
                    print(f"     -> Updating parent {parent} with grad {g}")
                    if isinstance(parent, Node):
                        parent.grad += g
                        print(f"     -> Node({parent._value:.2f}) grad updated to {parent.grad:.2f}")
            else:
                print(
                    f"  Skipping node: {node}, func: {node.func}, vjpmaker: {node._vjpmaker}, node_indices: {node.node_indices}")
        
        print(f"Final: Set root node gradient to {self.grad}")

    def zero_grad(self):
        self.grad = 0.0
        for parent in getattr(self, 'parents', []):
            if isinstance(parent, Node):
                parent.zero_grad()

    @staticmethod
    def _toposort(end_node, parents=operator.attrgetter("parents")):
        visited = set()
        order = []

        def dfs(n):
            if n in visited:
                return
            visited.add(n)
            for p in parents(n):
                if isinstance(p, Node):
                    dfs(p)
            order.append(n)

        dfs(end_node)
        # return order
        return reversed(order)



