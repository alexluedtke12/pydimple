#!/usr/bin/env python
# coding: utf-8

# Standard library imports
import warnings
import re
import operator # used to concisely define the operations in the PointwiseFunction class
from typing import List, Set, Optional, Any, Callable, Union


# Third-party library imports
import numpy as np
import pandas as pd   # dataframe environment used to store datasets
from scipy.stats import norm, iqr, multivariate_normal   # normal quantile used to construct confidence intervals
from scipy.integrate import nquad   # quadrature used in backward pass for density estimation
import sympy # used to determine if variables are zero based on the conditioning set
from statsmodels.nonparametric.kernel_density import KDEMultivariate, KDEMultivariateConditional # used in the density primitive
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.metrics import mean_squared_error # mean_squared_error function
import optuna.integration.lightgbm as lgb
from optuna import logging


def to_set(inp: Optional[List[Any]]) -> Set[Any]:
    """
    Convert a list or set to a set.

    :param inp: Input data that can be a list, set, or None.
    :type inp: Optional[List[Any]]
    :return: A set containing the elements from the input list or set. If the input is None, an empty set is returned.
    :rtype: Set[Any]
    :raises TypeError: If the input is not a list, set, or None.
    """
    if inp is None:
        return set()
    elif isinstance(inp, set):
        return inp
    elif isinstance(inp, list):
        return set(inp)
    else:
        raise TypeError("Unsupported type. Input must be a list, set, or None.")


class Singleton(type):
    """
    Singleton Metaclass.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Graph(metaclass=Singleton):
    """
    Computational graph. Consists of a set of Operators, Constants, and Distributions.

    :param num_folds: Number of folds to use in cross-fitting. Defaults to 5.
    :type num_folds: int
    :param simplify: A boolean indicating whether to use sympy to try to symbolically evaluate when a random variable is known to be zero based on the conditioning set. Defaults to True.
    :type simplify: bool
    """
    def __init__(self,num_folds=5,simplify=True): # Right now I'm communicating num_folds and simplify via the Graph class, so that they essentially serve as globals. E.g., I can access them via Graph().num_folds. But this isn't great practice, and I should probably instead pass them via the estimate function. Will do this later. Similarly for simplify.
        self.version = 0 # used to keep track of how many times clear() has been called (and ensure that Nodes from an old version of the Graph can't be used after clear is called)
        self.operators = set()
        self.constants = set()
        self.distributions = set()
        self.num_folds = num_folds
        self.simplify = simplify
        self.symbol_registry = {}  # Register for sympy symbols
        
    def clear(self):
        self.version += 1 # augment the version
        # Clear all sets
        self.operators.clear()
        self.constants.clear()
        self.distributions.clear()
        self.symbol_registry.clear()

        # Clear all nodes' values and adjoints
        for node_class in [Operator, Constant, Distribution]:
            for node in node_class.__subclasses__():
                self._reset_node(node)
                
        self.reset_counts(Node)
                
    def reset_counts(self, node):
        if hasattr(node, 'count'):
            node.count = 0
        else:
            for child in node.__subclasses__():
                self.reset_counts(child)

    def _reset_node(self, node):
        if hasattr(node, 'value'):
            node.value = None
        if hasattr(node, 'adjoint'):
            node.adjoint = None
        for child in node.__subclasses__():
            self._reset_node(child)


### Defining this will allow us to check if an object is a Graph node
class Node:
    def __init__(self):
        self.graph_version = Graph().version
    

### Distribution ###
class Distribution(Node):
    """
    A data-generating distribution node in the computational graph. This holds estimates of any needed features of the distribution (conditional means, densities, etc.).

    :param data: A DataFrame consisting of iid draws from the distribution.
    :type data: pd.DataFrame
    :param name: Name of the distribution. Defaults to "Dist/"+count.
    :type name: str, optional
    :param dtype: The type that the node holds, float, int, etc. Defaults to float.
    :type dtype: type, optional
    """
    count = 0
    def __init__(self, data, name=None, dtype=float):
        super().__init__()
        self.adjoint = None
        self.name = f'Dist/{Distribution.count}' if name is None else name
        Distribution.count += 1
        
        # Split the indices of the observations into 10 folds
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, Graph().num_folds)
        # Create an array of dicts
        splits = []
        for i in range(Graph().num_folds):
            # For each split, create a dictionary with train and validation indices
            val_indices = split_indices[i]
            train_indices = np.setdiff1d(indices, val_indices)

            splits.append({
                'train_inds': train_indices,
                'val_inds': val_indices
            })
            
        self.value = {"data": data,
                      "splits": splits,
                     }
        
        for i in range(Graph().num_folds):
            self.value[f"fold_{i}"] = {}
        
        # register symbols in sympy corresponding to each of the variables in the data DataFrame if Graph().simplify is true
        if Graph().simplify:
            # Fetch or define the variables as sympy symbols
            for name in data.columns:
                if name not in Graph().symbol_registry:
                    Graph().symbol_registry[name] = sympy.symbols(name)
        
    def __repr__(self):
        return f"Distribution: name:{self.name}, value:{self.value}"
        

class Constant(Node):
    """
    Represents a constant value in the computational graph.

    :param value: The value to assign to the constant. This value is immutable once set.
    :type value: Any
    :param name: An optional name for the constant. If not provided, a unique name is generated.
    :type name: str, optional
    :return: A new instance of Constant with the specified value and name.
    :rtype: Constant
    :raises ValueError: When attempting to modify the value of the constant after it is set.
    """
    count = 0
    def __init__(self, value, name=None):
        super().__init__()
        Graph().constants.add(self)
        self._value = value
        self.adjoint = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1
        
    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constant")
        
    def __call__(self, *args, **kwargs):
        if callable(self.value):
            return self.value(*args, **kwargs)
        else:
            return self.value


class Operator(Node):
    """
    An operator node in the computational graph.

    :param args: The input nodes to the operator.
    :type args: Node
    :param name: The name of the operator. Defaults to "Operator".
    :type name: str, optional
    """
    def __init__(self, *args, name='Operator'):
        super().__init__()
        Graph().operators.add(self)
        self.adjoint = None
        self.name = name
        self.requires_fold = False # set to True if the forward and backward arguments take fold as first argument. This is useful, for example, if using the function requires estimating an unknown nuisance using data
        
        # Check graph version
        if not all(isinstance(arg, Node) and arg.graph_version == Graph().version for arg in args):
            raise ValueError("All nodes in an operation must belong to the current graph.")
        
        # Define parents
        self.parents = list(args)
        self.value = None
    
    def __repr__(self):
        return f"Operator: name:{self.name}"
    
    def __call__(self, *args, **kwargs):
        if callable(self.value):
            return self.value(*args, **kwargs)
        else:
            return self.value


class PointwiseFunction:
    """
    Represents a function that applies a mathematical operation pointwise.

    :param func: A function.
    :type func: callable
    :param unknown_vars: A boolean indicating whether the variables the function relies on are known. If this is true, then the var_names argument is ignored.
    :type unknown_vars: bool
    :param var_names: A set or list of strings indicating the names of the variables the function relies on. Only used if unknown_vars is False. If a list is passed, it is converted to a set.
    :type var_names: set or list
    :param zero_if: Condition under which the function is a priori known to evaluate to zero.
    :type zero_if: str or sympy.Expr, optional

    Example:
        >>> f = PointwiseFunction(lambda x: x ** 2, unknown_vars=False, var_names=["x"])
    """
    def __init__(self, func, unknown_vars=True, var_names=None, zero_if = None):
        self.func = func
        self.unknown_vars = unknown_vars
        self.var_names = to_set(var_names)
        if Graph().simplify:
            # Fetch or define the variables as sympy symbols
            for name in self.var_names:
                if name not in Graph().symbol_registry:
                    Graph().symbol_registry[name] = sympy.symbols(name)
            self.set_zero_if(zero_if)
            self.zero_if_vars = {str(symbol) for symbol in self.zero_if.atoms() if symbol.is_Symbol}
        else:
            self.set_zero_if(None)

    def set_zero_if(self, condition):
        # Check if condition is a string
        if isinstance(condition, str):
            # Transform the string into a sympy expression
            condition = sympy.sympify(condition, locals=Graph().symbol_registry)
        elif condition is None:
            condition = sympy.S.false
        self.zero_if = condition

    # check if function will evaluate to zero based on one set of inputs
    def is_zero_one_input(self, **kwargs):
        if self.zero_if is None:
            return False
        else:
            # fetch symbols from Graph's symbol_registry
            symbols_mapping = {name: Graph().symbol_registry[name] for name in kwargs.keys()}
            condition = self.zero_if.subs(symbols_mapping).subs(kwargs)
            return not sympy.satisfiable(sympy.Not(condition))

    # check if function will evaluate to zero based on one row
    def is_zero_row(self, row):
        return self.is_zero_one_input(**row.to_dict())

    # check if function will evaluate to zero based on the information in each row of a dataframe and given the information in context.
    # context will typically be used to indicate a conditioning event
    # Note: the columns of this data frame may only contain a subset of the columns of self.var_names
    def is_zero(self, df):
        if self.zero_if == sympy.S.false: # if automatically true that an expression is zero, return a vector of Trues
            return pd.Series(False, index=df.index)
        elif self.zero_if == sympy.S.true: # if automatically true that an expression is zero, return a vector of Trues
            return pd.Series(True, index=df.index)
        else:
            # Define the subset of columns we're interested in
            cols_of_interest = list(set(df.columns) & self.zero_if_vars)          
            if len(cols_of_interest)==0:
                return pd.Series(False, index=df.index)
            else:
                # Create a new dataframe for unique rows
                unique_rows = df[cols_of_interest].drop_duplicates()

                # Apply self.is_zero_row function to each unique row and save the results to a dictionary
                # Each row is converted to a tuple to be used as a dictionary key
                results_dict = {tuple(row[cols_of_interest]): self.is_zero_row(row) for _, row in unique_rows.iterrows()}

                # Create a result series by mapping each row of the dataframe to its result
                return df[cols_of_interest].apply(lambda row: results_dict[tuple(row)], axis=1)
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    def log(self):
        with np.errstate(invalid='raise'):
            return PointwiseFunction(lambda *args, **kwargs: np.log(self.func(*args, **kwargs)),unknown_vars=self.unknown_vars,var_names=self.var_names) # POSSIBLE TO DO: set zero_if? But code will work fine without doing so

    def _binary_op(self, other, op, zero_if_fun = None):
        if isinstance(other, PointwiseFunction):
            return PointwiseFunction(lambda *args, **kwargs: op(self.func(*args, **kwargs), other.func(*args, **kwargs)),
                                     unknown_vars=self.unknown_vars or other.unknown_vars,
                                     var_names=self.var_names | other.var_names,
                                     zero_if=(None if (not Graph().simplify or not zero_if_fun) else sympy.simplify(zero_if_fun(self.zero_if,other.zero_if))))
        elif isinstance(other, (int, float)):
            return PointwiseFunction(lambda *args, **kwargs: op(self.func(*args, **kwargs), other),
                                     unknown_vars=self.unknown_vars,
                                     var_names=self.var_names,
                                     zero_if=(None if (not Graph().simplify or not zero_if_fun) else sympy.simplify(zero_if_fun(self.zero_if,(sympy.S.true if other==0 else sympy.S.false)))))
        elif isinstance(other, Node):
            return getattr(other, '__r' + op.__name__+'__')(self)
        else:
            raise TypeError(f"Unsupported operand type for {op.__name__}: 'PointwiseFunction' and '{type(other).__name__}'")
        
    def _reverse_op(self, other, op, zero_if_fun = None):
        if isinstance(other, (int, float)):
            return PointwiseFunction(lambda *args, **kwargs: op(other, self.func(*args, **kwargs)),
                                     unknown_vars=self.unknown_vars,
                                     var_names=self.var_names,
                                     zero_if=(None if (not Graph().simplify or not zero_if_fun) else sympy.simplify(zero_if_fun((sympy.S.true if other==0 else sympy.S.false),self.zero_if))))
        elif isinstance(other, Node):
            return getattr(other, '__' + op.__name__+'__')(self)
        else:
            raise TypeError(f"Unsupported operand type for {op.__name__}: '{type(other).__name__}' and 'PointwiseFunction'")
        
    def __add__(self, other):
        return self._binary_op(other, operator.add,zero_if_fun=sympy.And)

    def __radd__(self, other):
        return self.__add__(other,zero_if_fun=sympy.And)

    def __neg__(self):
        return PointwiseFunction(lambda *args, **kwargs: -self.func(*args, **kwargs),
                                 unknown_vars=self.unknown_vars,
                                 var_names=self.var_names,
                                 zero_if = self.zero_if)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub,zero_if_fun=sympy.And)

    def __rsub__(self, other):
        return self._reverse_op(other, operator.sub,zero_if_fun=sympy.And)

    def __mul__(self, other):
        return self._binary_op(other, operator.mul,zero_if_fun=sympy.Or)

    def __rmul__(self, other):
        return self._reverse_op(other, operator.mul,zero_if_fun=sympy.Or)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv,zero_if_fun=lambda x,y: x)

    def __rtruediv__(self, other):
        return self._reverse_op(other, operator.truediv,zero_if_fun=lambda x,y: x)

    def __pow__(self, other):
        return self._binary_op(other, operator.pow,zero_if_fun=lambda x,y: sympy.And(x, sympy.Not(y)))
    
        
class L2(PointwiseFunction):
    """
    Represents an L2 function, which is a subclass of PointwiseFunction
    with an associated Distribution object to represent the L2 space.

    :param func: A function or PointwiseFunction.
    :type func: callable or PointwiseFunction
    :param P: A Distribution object representing the L2 space.
    :type P: Distribution
    :param unknown_vars: A boolean indicating whether the variables the function relies on are known. If this is true, then the var_names argument is ignored. If `func` is a PointwiseFunction or L2, their `unknown_vars` values are used instead.
    :type unknown_vars: bool, optional
    :param var_names: A set or list of strings indicating the names of the variables the function relies on. Only used if unknown_vars is False. If a list is passed, it is converted to a set. If `func` is a PointwiseFunction or L2, their `var_names` values are used instead.
    :type var_names: set or list, optional
    :param zero_if: Condition under which the function is a priori known to evaluate to zero.
    :type zero_if: str or sympy.Expr, optional

    If `func` is already of type `L2`, the resulting `L2` object will inherit its `func` and `P` attributes. The value of `P` can be updated by providing a new value in the `P` argument of the constructor, overriding the previous value.

    Example:
        >>> f = L2(lambda x: x ** 2, P=P)  # P is an existing Distribution object
        >>> g = L2(f, P=Q)  # g inherits the func and var_names from f, but updates the value of P to an existing Distribution object Q
    """
    def __init__(self, func, P, unknown_vars=True, var_names=None, zero_if=None):
        if isinstance(func, L2) or isinstance(func, PointwiseFunction):
            # If func is already of type L2, this will just update the value of P. If func is a PointwiseFunction, this will use its attributs
            super().__init__(func.func, unknown_vars=func.unknown_vars, var_names=func.var_names, zero_if=func.zero_if)
        else:
            # Create a new PointwiseFunction with the given func
            super().__init__(func, unknown_vars=unknown_vars, var_names=var_names, zero_if=zero_if)
        self.P = P

    def __repr__(self):
        return f"L2({self.func}, P={self.P})"

def create_design_matrix(df, independent_vars):
    """
    Creates a matrix of independent variables from a pandas DataFrame.
    
    :param df: A pandas DataFrame containing the data.
    :type df: pd.DataFrame
    :param independent_vars: A string, or a list of strings, representing expressions for the independent variables. If None, returns None.
    :type independent_vars: str or list of str or None, optional
    :return: A pandas DataFrame that contains the independent variables, or None if independent_vars is None.
    :rtype: pd.DataFrame or None
    """
    if independent_vars is None:
        return None
    elif isinstance(independent_vars, str):
        independent_vars = [independent_vars]

    # Compute X
    X = df[independent_vars]

    return X


class add(Operator):
    """
    Addition operator node.

    :param a: The first input node.
    :type a: Node
    :param b: The second input node.
    :type b: Node
    :param name: The name of the addition operator. Defaults to 'add/{add.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, a, b, name=None):
        name = f'add/{add.count}' if name is None else name
        super().__init__(a, b, name=name)
        add.count += 1
        
    def forward(self, a, b):
        return a+b
    
    def backward(self, adj, a, b):
        return adj, adj
    
class subtract(Operator):
    """
    Subtraction operator node.

    :param a: The first input node.
    :type a: Node
    :param b: The second input node.
    :type b: Node
    :param name: The name of the subtraction operator. Defaults to 'sub/{subtract.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, a, b, name=None):
        name = f'sub/{subtract.count}' if name is None else name
        super().__init__(a, b, name=name)
        subtract.count += 1

    def forward(self, a, b):
        return a-b

    def backward(self, adj, a, b):
        return adj, -adj

class multiply(Operator):
    """
    Multiplication operator node.

    :param a: The first input node.
    :type a: Node
    :param b: The second input node.
    :type b: Node
    :param name: The name of the multiplication operator. Defaults to 'mul/{multiply.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, a, b, name=None):
        name = f'mul/{multiply.count}' if name is None else name
        super().__init__(a, b, name=name)
        multiply.count += 1
        
    def forward(self, a, b):
        return a*b
    
    def backward(self, adj, a, b):
        return b*adj, a*adj
    
class divide(Operator):
    """
    Division operator node.

    :param a: The first input node (numerator).
    :type a: Node
    :param b: The second input node (denominator).
    :type b: Node
    :param name: The name of the division operator. Defaults to 'div/{divide.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, a, b, name=None):
        name = f'div/{divide.count}' if name is None else name
        super().__init__(a, b, name=name)
        divide.count += 1
   
    def forward(self, a, b):
        return a/b
    
    def backward(self, adj, a, b):
        return adj/b, -adj*a/(b**2)
    
class power(Operator):
    """
    Power operator node.

    :param a: The first input node (base).
    :type a: Node
    :param b: The second input node (exponent).
    :type b: Node
    :param name: The name of the power operator. Defaults to 'pow/{power.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, a, b, name=None):
        name = f'pow/{power.count}' if name is None else name
        super().__init__(a, b, name=name)
        self.b_is_constant = isinstance(b,Constant)
        power.count += 1
   
    def forward(self, a, b):
        return a**b
    
    def backward(self, adj, a, b):
        if not self.b_is_constant: # if b is a constant then there's no need to backpropagate anything to it, and also no need to do error handling
            if isinstance(a,PointwiseFunction):
                out2 = adj*a.log()*(a**b)
            else:
                with np.errstate(invalid='raise'): # throw error if np.log(0) or np.log(negative number)
                    out2 = adj*np.log(a)*(a**b)
        else:
            out2 = 0*adj
        return adj*b*(a**(b-1)), out2
    
class log(Operator): # natural logarithm
    """
    Natural logarithm operator node.

    :param a: The input node.
    :type a: Node
    :param name: The name of the logarithm operator. Defaults to 'log/{log.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, a, name=None):
        name = f'log/{log.count}' if name is None else name
        super().__init__(a, name=name)
        log.count += 1
        
    def forward(self, a):
        return a.log()
    
    def backward(self, adj, a):
        return adj/a,



def binary_operation(func, self, other):
    """
    Performs a binary operation between a node and another object.

    :param func: The binary operation function to apply.
    :type func: callable
    :param self: The node on which the binary operation is performed.
    :type self: Node
    :param other: The other object involved in the binary operation.
    :type other: Node, float, int, or PointwiseFunction
    :return: The result of applying the binary operation.
    :rtype: Node
    :raises TypeError: If the type of `other` is not supported for the binary operation.
    """
    if isinstance(other, Node): # if both self and other are nodes, then can apply the function
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int) or isinstance(other, PointwiseFunction): # if other isn't a node, then introduce a constant into the graph and then apply the function
        return func(self, Constant(other))
    raise TypeError(f"Cannot apply {func.__name__} to a node and an object of type {type(other).__name__}.")

def reverse_binary_operation(func, self, other):
    """
    Performs a binary operation between a node and another object, with the operands reversed.

    :param func: The binary operation function to apply.
    :type func: callable
    :param self: The node on which the binary operation is performed.
    :type self: Node
    :param other: The other object involved in the binary operation.
    :type other: Node, float, int, or PointwiseFunction
    :return: The result of applying the binary operation with the operands reversed.
    :rtype: Node
    :raises TypeError: If the type of `other` is not supported for the binary operation.
    """
    # same as binary_operation except the order of the operands is reversed
    if isinstance(other, Node):
        return func(other, self)  # reverse the order of operands
    if isinstance(other, float) or isinstance(other, int) or isinstance(other, PointwiseFunction):
        return func(Constant(other), self)  # reverse the order of operands
    raise TypeError(f"Cannot apply {func.__name__} to an object of type {type(other).__name__} and a node.")
    
def generate_binary_operation_doc(return_desc):
    return f"""
    :param self: First argument to binary operation
    :type self: Node
    :param other: Second argument to binary operation
    :type other:  Node, float, int, or PointwiseFunction
    :return: {return_desc}  (pointwise operation)
    :rtype: Node
    """

Node.__add__ = lambda self, other: binary_operation(add, self, other)
Node.__add__.__doc__ = generate_binary_operation_doc("self+other")
Node.__radd__ = lambda self, other: reverse_binary_operation(add, self, other)
Node.__radd__.__doc__ = generate_binary_operation_doc("other+self")
Node.__sub__ = lambda self, other: binary_operation(subtract, self, other)
Node.__sub__.__doc__ = generate_binary_operation_doc("self-other")
Node.__rsub__ = lambda self, other: reverse_binary_operation(subtract, self, other)
Node.__rsub__.__doc__ = generate_binary_operation_doc("other-self")
Node.__mul__ = lambda self, other: binary_operation(multiply, self, other)
Node.__mul__.__doc__ = generate_binary_operation_doc("self*other")
Node.__rmul__ = lambda self, other: reverse_binary_operation(multiply, self, other)
Node.__rmul__.__doc__ = generate_binary_operation_doc("other*self")
Node.__truediv__ = lambda self, other: binary_operation(divide, self, other)
Node.__truediv__.__doc__ = generate_binary_operation_doc("self/other")
Node.__rtruediv__ = lambda self, other: reverse_binary_operation(divide, self, other)
Node.__rtruediv__.__doc__ = generate_binary_operation_doc("other/self")
Node.__pow__ = lambda self, other: binary_operation(power, self, other)
Node.__pow__.__doc__ = generate_binary_operation_doc("self**other")
Node.__rpow__ = lambda self, other: reverse_binary_operation(power, self, other)
Node.__rpow__.__doc__ = generate_binary_operation_doc("other**self")
Node.__neg__ = lambda self: binary_operation(multiply, self, Constant(-1))
Node.__neg__.__doc__ = """
    :param self: Node to negate
    :type self: Node
    :return: -self  (pointwise operation)
    :rtype: Node
    """


def fit_lightgbm_model(X_train, 
                       y_train, 
                       X_val, 
                       y_val,
                       objective="regression", 
                       metric="rmse",
                       custom_obj=None,
                       custom_eval=None,
                       weights_train=None,
                       weights_val=None,
                       alpha=None):
    """
    Fit a LightGBM model to the given data, optionally using a custom objective function and a custom evaluation function.

    :param X_train: Feature training data.
    :type X_train: pandas.DataFrame
    :param y_train: Target training data.
    :type y_train: pandas.Series or numpy.ndarray
    :param X_val: Feature validation data.
    :type X_val: pandas.DataFrame
    :param y_val: Target validation data.
    :type y_val: pandas.Series or numpy.ndarray
    :param objective: Objective for LightGBM model, default is "regression".
    :type objective: str, optional
    :param metric: Metric for LightGBM model, default is "rmse".
    :type metric: str, optional
    :param custom_obj: Optional custom objective function for training.
    :type custom_obj: callable, optional
    :param custom_eval: Optional custom evaluation function for validation.
    :type custom_eval: callable, optional
    :param weights_train: Optional instance weights for training data.
    :type weights_train: pandas.Series or numpy.ndarray, optional
    :param weights_val: Optional instance weights for validation data.
    :type weights_val: pandas.Series or numpy.ndarray, optional
    :param alpha: Optional alpha parameter for the objective function.
    :type alpha: float, optional
    :return: Trained LightGBM Booster object.
    :rtype: lightgbm.Booster
    """
    # Define base parameters
    params = {
        "objective": objective,
        "alpha": alpha,
        "metric": metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    logging.set_verbosity(logging.WARNING)

    # Create datasets for LightGBM, incorporating any weights if provided
    train_data = lgb.Dataset(X_train, y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, y_val, weight=weights_val)

    # Instantiate tuner and perform tuning
    tuner = lgb.LightGBMTuner(params, train_data,
                              valid_sets=[val_data],
                              fobj=custom_obj, feval=custom_eval,
                              show_progress_bar=False,
                              callbacks=[lgb.early_stopping(10, verbose=False)])

    # Run the tuning
    tuner.run()

    # Use the best model
    fit = tuner.get_best_booster()

    return fit


############################################
#####          Embedding class         #####
############################################ 
# TO DO/CAUTION: need to implement this for embeddings from L^2(Q_X) into L^2(P_X)? Or will all objects by default be lifted to L^2(Q)/L^2(P) by other primitives
class embed(Operator):
    """
    Embed an element f of L^2(Q) into L^2(P).

    :param f: The function to be embedded, an element of L^2(Q).
    :type f: L2
    :param P: The distribution representing the target L^2 space.
    :type P: Distribution
    :param name: The name of the embedding operator. Defaults to 'embed/{embed.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, f, P, name=None):
        name = f'embed/{embed.count}' if name is None else name
        super().__init__(f, name=name)
        
        if not isinstance(P,Distribution):
            raise TypeError("P must be a Distribution object.")
        self.P = P
        self.requires_fold = True
                
        embed.count += 1
        
    # custom objective that can be fed into lightgbm to learn dQ/dP
    def density_ratio_train_objective(self, preds,train_data):
        S = train_data.get_label()
        return np.where(S == 1, 2*preds, -2), np.where(S == 1, 2, 0) # gradient, hessian

    # custom objective that can be fed into lightgbm to evaluate estimate of dQ/dP
    def density_ratio_eval_objective(self, preds,eval_data):
        S = eval_data.get_label()
        return 'IntSqErr', np.mean(np.where(S == 1, preds**2, -2*preds)), False # eval_name, eval_result, is_higher_better

    def forward(self, fold, f): # including fold input since this function takes as input a distribution (which relies on unknowns)
        if not isinstance(f,L2):
            raise TypeEror("f must be an L2 object.")
        self.Q = f.P
        
        P_cols = self.P.value['data'].columns
        Q_cols = self.Q.value['data'].columns
        if f.unknown_vars:
            if not set(Q_cols).issubset(P_cols): # throw error if all columns in Q's data are not contained in P's data
                raise ValueError("If f.unknown_vars is False, then the column names of the dataframe for Q must be a subset of the column names of the dataset for P.")
            else:
                self.var_names = Q_cols.tolist()
        else:
            if not f.var_names.issubset(P_cols): # throw error if column names of P are not contained in f.var_names
                raise ValueError("If f.unknown_vars is True, then the column names of the dataframe for P must be a subset of f.var_names.")
            else:
                self.var_names = f.var_names
            
        return L2(f.func,P=self.P,unknown_vars=False,var_names=self.var_names) # same as input except changed the distribution
    
    def backward(self, fold, adj, f): # including fold input since this function takes as input a distribution (which relies on unknowns)
        Q_train_inds = self.Q.value['splits'][fold]['train_inds']
        Q_val_inds = self.Q.value['splits'][fold]['val_inds']
        P_train_inds = self.P.value['splits'][fold]['train_inds']
        P_val_inds = self.P.value['splits'][fold]['val_inds']
        
        X_train = pd.concat([self.Q.value['data'][list(self.var_names)].iloc[Q_train_inds],
                             self.P.value['data'][list(self.var_names)].iloc[P_train_inds]])
        X_val = pd.concat([self.Q.value['data'][list(self.var_names)].iloc[Q_val_inds],
                           self.P.value['data'][list(self.var_names)].iloc[P_val_inds]])
        y_train = np.concatenate((np.ones(len(Q_train_inds)), np.zeros(len(P_train_inds))))
        y_val = np.concatenate((np.ones(len(Q_val_inds)), np.zeros(len(P_val_inds))))

        fit = fit_lightgbm_model(X_train, y_train, X_val, y_val, 
                                 objective="custom",
                                 metric='IntSqErr',
                                 custom_obj=self.density_ratio_train_objective,
                                 custom_eval=self.density_ratio_eval_objective)


        density_ratio = PointwiseFunction(lambda x: fit.predict(x[list(self.var_names)], num_iteration=fit.best_iteration),unknown_vars=False,var_names=self.var_names)

        return density_ratio*adj, # POSSIBLE TO DO: coerce to an L2 object?


def add_unique_column(df, new_col_name, new_col_values):
    """
    Add a new column with a unique name to a DataFrame.

    If the proposed new column name already exists in the DataFrame, 
    this function will append a number to it to make it unique.

    :param df: The DataFrame to add the new column to.
    :type df: pandas.DataFrame
    :param new_col_name: The proposed name of the new column.
    :type new_col_name: str
    :param new_col_values: The values for the new column.
    :type new_col_values: list or numpy.ndarray or pandas.Series
    :return: A tuple containing the modified DataFrame and the actual name of the new column.
    :rtype: tuple
    """

    original_col_name = new_col_name
    counter = 1

    # Check if the proposed column name is in the DataFrame's columns
    while new_col_name in df.columns:
        # If it is, append a counter to the column name and increment the counter
        new_col_name = original_col_name + str(counter)
        counter += 1

    # Add the new column to the DataFrame
    df[new_col_name] = new_col_values

    return df, new_col_name


def convert_to_sympy(condition):
    """
    Convert a string condition to a sympy equation or inequality.

    This function takes a condition as a string, splits it into left and right parts
    based on the operator, and then returns a corresponding sympy equation or inequality.

    If no operator is found in the condition, it treats the condition as an expression.

    :param condition: A string representing an equation or inequality.
    :type condition: str
    :return: A sympy equation or inequality representing the input condition.
    :rtype: sympy.Eq or sympy.Relational or sympy.Expr

    Example:
        >>> convert_to_sympy("x == 5")
        Eq(x, 5)
        >>> convert_to_sympy("x > 5")
        x > 5
        >>> convert_to_sympy("x + y")
        x + y
    """
    
    # Dictionary to map string operator to its corresponding sympy function
    operator_map = {"==": sympy.Eq, ">": sympy.Gt, "<": sympy.Lt, 
                    ">=": sympy.Ge, "<=": sympy.Le, "!=": sympy.Ne}
    
    # Split the condition into left and right parts and operator
    for operator, sympy_operator in operator_map.items():
        if operator in condition:
            left_str, right_str = condition.split(operator)
            # Parse the left and right string into sympy objects
            left = sympy.parse_expr(left_str)
            right = sympy.parse_expr(right_str)
            # Return the sympy equation/inequality
            return sympy_operator(left, right)
    
    # If no operator is found, treat the condition as an expression
    return sympy.S(condition)


def implicit_lift(fold,adj,P,which_vars):
    """
    Implicitly lift a primitive from L^2(P_X) to L^2(P), where P_X is the marginal distribution
    of a subvector X of Z~P.

    :param fold: Current fold in the backward pass.
    :type fold: int
    :param adj: Adjoint value supplied in the backward pass.
    :type adj: PointwiseFunction or float
    :param P: Distribution supplied in the backward pass.
    :type P: Distribution
    :param which_vars: Indices or names of variables in X.
    :type which_vars: list or set
    :return: An estimate of E[adj(Z)|X]. When the adj supplied to a primitive is replaced by adj_proj,
             the output of that primitive is implicitly lifted to L^2(P).
    :rtype: PointwiseFunction
    """
    fold_name = 'fold_'+str(fold)
    train_inds, val_inds = P['splits'][fold]['train_inds'], P['splits'][fold]['val_inds']
    which_vars_set = to_set(which_vars)
    
    if isinstance(adj, PointwiseFunction) and (adj.unknown_vars or not adj.var_names.issubset(which_vars_set)):
        # estimated adj values at observations in dataset
        adj_preds = adj(P['data'])
        dat = create_design_matrix(P['data'],list(which_vars_set))
        fit_adj_proj = fit_lightgbm_model(dat.iloc[train_inds], np.asarray(adj_preds)[train_inds], dat.iloc[val_inds], np.asarray(adj_preds)[val_inds])
        return PointwiseFunction(lambda x: fit_adj_proj.predict(create_design_matrix(x, list(which_vars_set)), num_iteration=fit_adj_proj.best_iteration),unknown_vars=False,var_names=which_vars_set)
    else:
        return adj

def RV(rv_str):
    """
    Create a PointwiseFunction that extracts a random variable from a DataFrame.

    :param rv_str: The name of the random variable (column) to extract.
    :type rv_str: str
    :return: A PointwiseFunction that takes a DataFrame as input and returns the column
             specified by `rv_str`.
    :rtype: PointwiseFunction

    Example:
        >>> df = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
        >>> X = RV('X')
        >>> X(df)
        0    1
        1    2
        2    3
        Name: X, dtype: int64
    """
    return PointwiseFunction(lambda df: df[rv_str], unknown_vars=False, var_names=[rv_str])

class E(Operator):
    """
    Marginal or conditional mean of draws from a univariate distribution.

    :param P: A distribution.
    :type P: Distribution
    :param dep: Either a string containing the name of a column in the dataset in P, or a Node/PointwiseFunction.
    :type dep: str or Node or PointwiseFunction
    :param indep_vars: Variables to regress against.
    :type indep_vars: list of str or None, optional
    :param fixed_vars: A set of conditions expressed as strings. e.g., {'A1==1', 'A2==A1', 'A3*A2<A1+A4'}.
                       The conditions should be valid Python expressions and should only involve columns present in P['data'].
    :type fixed_vars: set of str or None, optional
    :param name: The name of the operator. Defaults to 'E/{E.count}'.
    :type name: str, optional
    """
    count = 0
    def __init__(self, P, dep, indep_vars=None, fixed_vars=None, name=None):
        name = f'E/{E.count}' if name is None else name
        if isinstance(dep, str):
            self.dep_str = dep
            dep = RV(self.dep_str)
        if not isinstance(dep,Node):
            dep = Constant(dep)
        super().__init__(P, dep, name=name)
        self.P = P
        self.indep_vars = [indep_vars] if isinstance(indep_vars,str) else indep_vars
        self.fixed_vars = fixed_vars
        self.fixed_var_names = self.extract_variable_names(fixed_vars, P.value['data'].columns.tolist())
        self.requires_fold = True
        self.simplify = Graph().simplify
        if self.simplify:
            self.fixed_vars_sympy = sympy.S.true if fixed_vars is None else sympy.And(*[convert_to_sympy(condition) for condition in fixed_vars])
            
        E.count += 1

    def __call__(self, x=None):
        # When this object is called as a function, it will return the saved forward result
        if callable(self.saved_forward):
            return self.saved_forward(x)
        else:
            return self.saved_forward
        
    def extract_variable_names(self, conditions, df_cols):
        variable_names = set()
        if conditions:
            # Iterate over the columns from longest to shortest
            df_cols_sorted = sorted(df_cols, key=len, reverse=True)
            for condition in conditions:
                for col in df_cols_sorted:
                    # Use word boundary regex to ensure exact match
                    if re.search(r'\b' + col + r'\b', condition):
                        variable_names.add(col)
        return variable_names
        
    def create_fixed_var(self, df):
        mask = pd.Series([True]*len(df), index=df.index)
        for condition in self.fixed_vars:
            mask &= df.query(condition).index.to_series().reindex(mask.index, fill_value=False)
        return mask
        
    def get_indices(self, P, fold, fixed_only=True, exclude=None): # gets training and validation indices for the given fold. Only returns those for which the conditions in fixed_vars are satisfied if fixed_only is True. And automatically omits all the indices for which exclude is True
        train_mask = P['data'].index.isin(P['splits'][fold]['train_inds'])
        val_mask = P['data'].index.isin(P['splits'][fold]['val_inds'])
        
        if fixed_only and self.fixed_vars is not None:
            fixed_var = self.create_fixed_var(P['data'])
            train_mask = train_mask & fixed_var
            val_mask = val_mask & fixed_var
            
        if exclude is not None:
            train_mask = train_mask & ~exclude
            val_mask = val_mask & ~exclude

        train_inds = P['data'][train_mask].index.tolist()
        val_inds = P['data'][val_mask].index.tolist()

        return train_inds, val_inds
    
    def encode_label(self,E_adj,fixed_var):
        # Get the min and max values of E_adj
        min_val = E_adj.min()
        max_val = E_adj.max()

        # Calculate scale to map E_adj to the range [0, 1)
        # If min_val equals max_val, set scale to 1 and shift to 0
        if min_val != max_val:
            self.E_adj_scale = 1 / (max_val - min_val)
            self.E_adj_shift = -min_val
        else:
            self.E_adj_scale = 1
            self.E_adj_shift = -min_val

        # Apply scale and shift
        E_adj_transformed = (E_adj + self.E_adj_shift) * self.E_adj_scale
        
        return fixed_var + E_adj_transformed
    
    def decode_label(self,encoded_label):
        fixed_var = np.floor(encoded_label)  # Get the integer part
        E_adj = encoded_label - fixed_var  # Extract E_adj
        # Reverse the transformation for E_adj
        E_adj = E_adj / self.E_adj_scale - self.E_adj_shift
        return fixed_var, E_adj
        
    # custom objective that can be fed into lightgbm to learn XX
    def ipw_train_objective(self, preds,train_data):
        fixed_var, E_adj = self.decode_label(train_data.get_label())
        return np.where(fixed_var == 1, 2*preds-2*E_adj, -2*E_adj), np.where(fixed_var == 1, 2, 0) # gradient, hessian

    # custom objective that can be fed into lightgbm to evaluate estimate of XX
    def ipw_eval_objective(self, preds,eval_data):
        fixed_var, E_adj = self.decode_label(eval_data.get_label())
        return 'IntSqErr', np.mean(np.where(fixed_var == 1, preds**2-2*preds*E_adj, -2*preds*E_adj)), False # eval_name, eval_result, is_higher_better
    
    def forward(self, fold, P, dep): # including fold input since this function takes as input a distribution (which relies on unknowns)
        fold_name = 'fold_'+str(fold)
        if self.name not in P[fold_name]:
            y, X = dep(P['data']), create_design_matrix(P['data'],self.indep_vars)
            train_inds, val_inds = self.get_indices(P, fold)
            if self.indep_vars is None:
                P[fold_name][self.name] = {0: np.mean(y[train_inds])} # If np.asarray(y) the mean doesn't exist, put it in the first slot
            else:
                # regress dependent variable against X
                fit = fit_lightgbm_model(X.iloc[train_inds], np.asarray(y)[train_inds], X.iloc[val_inds], np.asarray(y)[val_inds])
                
                P[fold_name][self.name] = {0: L2(lambda x: fit.predict(create_design_matrix(x, self.indep_vars), num_iteration=fit.best_iteration),P=self.P,unknown_vars=False,var_names=self.indep_vars)} # If the conditional mean doesn't exist, put it in the first slot

            self.saved_forward = P[fold_name][self.name][0] # This needs to be updated! The current code just makes the callable version of this function use whatever was saved last, which may cause bugs down the line if there are ever cross-fold calculations
        return P[fold_name][self.name][0]
    
    def backward(self, fold, adj, P, dep): # including fold input since this function takes as input a distribution (which relies on unknowns)
        fold_name = 'fold_'+str(fold)
               
        if isinstance(adj, (float, int)):
            adj_fun = PointwiseFunction(lambda z: np.full(z.shape[0], adj), unknown_vars = False, var_names=None, zero_if=(sympy.S.true if adj==0 else sympy.S.false))
        else:
            adj_fun = adj
            
        # project adj onto L^2(P_X)
        # CAUTION: in the current implementation, if adj is projected then all is_zero information is lost (!!)
        #          this is ok but may hurt performance of the estimator somewhat
        #          (worse finite-sample performance and/or stronger conditions on nuisance estimators for asymptotic linearity)
        adj_proj = implicit_lift(fold,adj_fun,P,self.indep_vars)
            
        # if fixed_vars is not None, then add estimated inverse probability weight in front of adj
        if self.fixed_vars is not None:
            encoded_label = self.encode_label(np.asarray(adj_proj(P['data'])),self.create_fixed_var(P['data']).astype(int))
            X = create_design_matrix(P['data'],self.indep_vars)
            train_inds, val_inds = self.get_indices(P, fold, fixed_only=False, exclude=(None if not self.simplify else adj_proj.is_zero(X)))
            fit = fit_lightgbm_model(X.iloc[train_inds], np.asarray(encoded_label)[train_inds], X.iloc[val_inds], np.asarray(encoded_label)[val_inds], 
                         objective="custom", metric='IntSqErr',
                         custom_obj=self.ipw_train_objective, custom_eval=self.ipw_eval_objective)
            def adj_ipw_fun(z):
                design_mat = create_design_matrix(z, self.indep_vars)
                if self.simplify:
                    adj_nonzero = 1-adj_proj.is_zero(design_mat).astype(int).values
                else:
                    adj_nonzero = 1
                return adj_nonzero * self.create_fixed_var(z).astype(int).values * fit.predict(design_mat, num_iteration=fit.best_iteration)
            adj_ipw = PointwiseFunction(adj_ipw_fun,unknown_vars=False,var_names=to_set(self.indep_vars) | self.fixed_var_names, zero_if=(None if not self.simplify else sympy.simplify(sympy.Or(adj_proj.zero_if,sympy.Not(self.fixed_vars_sympy)))))
        else:
            adj_ipw = adj_proj
            
        def residual(z):
            y, X = dep(z), create_design_matrix(z,self.indep_vars)
            return (y-(P[fold_name][self.name][0] if self.indep_vars is None else P[fold_name][self.name][0](X)))
        residual = PointwiseFunction(residual,unknown_vars=False,var_names=to_set(self.indep_vars) | dep.var_names)
                
        return adj_ipw*residual, adj_ipw


def Var(P, dep, indep_vars=None, fixed_vars=None): # TO DO: implement this as an Operator rather than as a wrapper
    """
    Marginal or conditional variance operator, defined as a wrapper for E.

    :param P: A distribution.
    :type P: Distribution
    :param dep: Either a string containing the name of a column in the dataset in P, or a Node/PointwiseFunction.
    :type dep: str or Node or PointwiseFunction
    :param indep_vars: Variables to regress against.
    :type indep_vars: list of str or None, optional
    :param fixed_vars: A set of conditions expressed as strings. e.g., {'A1==1', 'A2==A1', 'A3*A2<A1+A4'}.
                       The conditions should be valid Python expressions and should only involve columns present in P['data'].
    :type fixed_vars: set of str or None, optional
    :return: The marginal or conditional variance.
    :rtype: Operator
    """
    if isinstance(dep,str):
        dep_str = dep
        dep_sq = PointwiseFunction(lambda df: df[dep_str]**2, unknown_vars=False, var_names=[dep_str])
    else:
        dep_sq = dep**2
    return E(P,dep_sq,indep_vars=indep_vars,fixed_vars=fixed_vars,name=None) - E(P,dep,indep_vars=indep_vars,fixed_vars=fixed_vars,name=None)**2


class Density(Operator):
    """
    Marginal or conditional density function of a real-valued dependent variable.

    :param P: A distribution.
    :type P: Distribution
    :param dep_vars: Dependent variables.
    :type dep_vars: list of str
    :param indep_vars: Independent variables.
    :type indep_vars: list of str or None, optional
    :param indep_type: Either None or a string the length of indep_vars, with the string specifying a type for each variable
                       (c: Continuous, u: Unordered, o: Ordered). E.g., indep_type='ccuo'. If not provided, the type is assumed to be continuous.
    :type indep_type: str or None, optional
    :param fixed_vars: A set of conditions expressed as strings. e.g., {'A1==1', 'A2==A1', 'A3*A2<A1+A4'}.
                       The conditions should be valid Python expressions and should only involve columns present in P['data'].
    :type fixed_vars: set of str or None, optional
    :param name: The name of the operator. Defaults to 'density/{Density.count}'.
    :type name: str, optional
    :param verbose: Returns warnings if True.
    :type verbose: bool, optional
    """
    count = 0
    def __init__(self, P, dep_vars, indep_vars=None, indep_type=None, name=None, verbose=True):
        name = f'density/{Density.count}' if name is None else name
        super().__init__(P, name=name)
        
        self.P = P
        
        self.dep_vars = [dep_vars] if isinstance(dep_vars,str) else dep_vars
        self.indep_vars = [indep_vars] if isinstance(indep_vars,str) else indep_vars
        if dep_vars is not None:
            self.dep_type = 'c'*len(self.dep_vars)
        if indep_vars is not None:
            self.indep_type = 'c'*len(self.indep_vars) if indep_type is None else indep_type
        self.requires_fold = True
        
        # will store bandwidths from KDE fits to be used in the backward pass
        self.bws = {}

        Density.count += 1
        
    def forward(self, fold, P): # including fold input since this function takes as input a distribution (which relies on unknowns)
        fold_name = 'fold_'+str(fold)
        if self.name not in P[fold_name]:
            y, X = create_design_matrix(P['data'],self.dep_vars), create_design_matrix(P['data'],self.indep_vars)
            train_inds, val_inds = P['splits'][fold]['train_inds'], P['splits'][fold]['val_inds']
            if self.indep_vars is None:
                fit = KDEMultivariate(data=y.iloc[train_inds], var_type=self.dep_type, bw='normal_reference')
                P[fold_name][self.name] = {0: L2(lambda x: fit.pdf(data_predict=create_design_matrix(x, self.dep_vars)),P=self.P,unknown_vars=False,var_names=self.dep_vars)} # If the density doesn't exist, put it in the first slot
            else:
                fit = KDEMultivariateConditional(endog=y.iloc[train_inds], exog=X.iloc[train_inds], dep_type=self.dep_type, indep_type=self.indep_type,bw='normal_reference')
                P[fold_name][self.name] = {0: L2(lambda x: fit.pdf(endog_predict=create_design_matrix(x, self.dep_vars), exog_predict=create_design_matrix(x, self.indep_vars)),P=self.P,unknown_vars=False,var_names=self.dep_vars+self.indep_vars)} # If the conditional density doesn't exist, put it in the first slot
            self.bws[fold_name] = fit.bw # used on the backward pass to take expectations wrt KDE
                
            self.saved_forward = P[fold_name][self.name][0] # This needs to be updated! The current code just makes the callable version of this function use whatever was saved last, which may cause bugs down the line if there are ever cross-fold calculations
        return P[fold_name][self.name][0]

    def backward(self, fold, adj, P): # including fold input since this function takes as input a distribution (which relies on unknowns)
        fold_name = 'fold_'+str(fold)
        train_inds, val_inds = P['splits'][fold]['train_inds'], P['splits'][fold]['val_inds']
        
        all_vars = to_set(self.dep_vars).union(to_set(self.indep_vars))
        
        # if adj depends on variables other than those in the union of dep_vars and indep_vars, then
        # we implicitly insert a lifting operation between this density primitive and the subsequent ones.
        # this is achieved by regressing adj against the variables in the union of dep_vars and indep_vars
        adj_proj = implicit_lift(fold,adj,P,all_vars)
                
        out = P[fold_name][self.name][0] * adj_proj
        
        if self.indep_vars is None:
            # approximate mean of out under density from forward pass using Monte Carlo samples from KDE
            # Number of samples per row of y
            n_mc = 50
            y = create_design_matrix(P['data'].iloc[train_inds],self.dep_vars)
            # Monte Carlo draws from forward pass density estimate
            y_mc = pd.DataFrame(
                np.repeat(y.to_numpy(), n_mc, axis=0) + np.random.multivariate_normal(np.zeros(y.shape[1]), np.diag(self.bws[fold_name]**2), y.shape[0]*n_mc),
                columns=y.columns)
            out = out - np.mean(out(y_mc))
        else:
            n_mc = 2000 if len(self.dep_vars)>1 else 1000
            dat_mc = pd.concat([P['data'].iloc[train_inds]] * n_mc, ignore_index=True)
            if len(self.dep_vars)>1: # Monte Carlo approximation
                for i in range(len(self.dep_vars)):
                    dep_var=self.dep_vars[i]
                    dat_mc[dep_var] += np.random.normal(0.,self.bws[fold_name][i],dat_mc.shape[0])
            else: # quasi-Monte Carlo
                my_grid = np.linspace(1 / (2 * n_mc), 1 - 1 / (2 * n_mc), n_mc) # uniform grid from 0 to 1
                quasi_mc = norm.ppf(my_grid,0.,self.bws[fold_name][0]) # quasi-normal draws along this uniform grid
                dat_mc[self.dep_vars[0]] += np.repeat(quasi_mc, len(train_inds)) # add these repetitions to dat_mc
            fit = KernelReg(endog = out(dat_mc), exog=create_design_matrix(dat_mc,self.indep_vars), var_type=self.indep_type,reg_type='lc', bw=self.bws[fold_name][-len(self.indep_vars):])
            out = out - PointwiseFunction(lambda x: fit.fit(data_predict=create_design_matrix(x, self.indep_vars))[0],unknown_vars=False,var_names=self.indep_vars)

        return out, 


def topological_sort(estimand):
    """
    Perform a topological sort on the nodes in the computational graph.

    :param estimand: The node to start the topological sort from.
    :type estimand: Node
    :return: A topologically sorted list of nodes in the computational graph.
    :rtype: list of Node
    """
    # depth-first search to produce a topological ordering of the nodes
    already_visited = set()
    order = [] # will be updated to contain the topological ordering
    
    def visit(node):
        if node not in already_visited:
            already_visited.add(node)
            if isinstance(node, Operator):
                for parent in node.parents:
                    visit(parent)
            order.append(node)
    
    # perform depth-first search via the visit function below
    visit(estimand)
    # now order contains a topological ordering of the nodes

    return order


def forward_pass(order, fold):
    """
    Construct an initial estimate by performing a forward pass on the topologically sorted nodes.

    :param order: A topologically sorted list of nodes in the computational graph.
    :type order: list of Node
    :param fold: An integer specifying the fold to use for estimating nuisances.
    :type fold: int
    :return: The initial estimate obtained from the forward pass.
    :rtype: Any
    """

    for node in order:
        if isinstance(node, Operator):
            if node.requires_fold:
                node.value = node.forward(fold, *[prev_node.value for prev_node in node.parents])
            else:
                node.value = node.forward(*[prev_node.value for prev_node in node.parents])

    return order[-1].value
    
 
def backward_pass(order, fold, init=1):
    """
    Compute the efficient influence operator by performing a backward pass on the topologically sorted nodes.

    :param order: A topologically sorted list of nodes in the computational graph.
    :type order: list of Node
    :param fold: An integer specifying the fold to use for estimating nuisances.
    :type fold: int
    :param init: The initial value for the adjoint of the last node in the order. Defaults to 1.
    :type init: Any, optional
    :return: None, but the adjoint information in nodes is updated.
    :rtype: None
    """
    alread_visited = set()
    order[-1].adjoint = init
    for node in reversed(order):
        if isinstance(node, Operator):
            parents = node.parents
            if node.requires_fold:
                adjs = node.backward(fold, node.adjoint, *[x.value for x in parents])
            else:
                adjs = node.backward(node.adjoint,*[x.value for x in parents])
                
            for inp, adj in zip(parents, adjs):
                if inp not in alread_visited:
                    inp.adjoint = adj
                else:
                    inp.adjoint += adj
                alread_visited.add(inp)
                
    return None


##########################################################
#############    Estimate the specified node  ############
##########################################################
def estimate(estimand,level=0.95,ci_type='wald',num_boot=2000):
    """
    Estimate the value of the specified node and compute a confidence interval.

    :param estimand: The node whose value is to be estimated.
    :type estimand: Node
    :param level: The nominal level of the reported confidence interval. Defaults to 0.95.
    :type level: float, optional
    :param ci_type: The type of confidence interval to compute. One of 'bca', 'percentile_boot', or 'wald'. Defaults to 'wald'.
    :type ci_type: str, optional
    :param num_boot: The number of bootstrap replicates to use to compute the standard error. Ignored if ci_type is 'wald'. Defaults to 2000.
    :type num_boot: int, optional
    :return: A dictionary containing the estimate of the node ('est'), the standard error estimate ('se'), and the confidence interval ('ci').
    :rtype: dict
    :raises ValueError: If an invalid ci_type is provided.
    
    .. caution::
       This function automatically clears the computation graph after being called!
    """

    order = topological_sort(estimand)
    
    ests = []
    uncertainty = [] # will either hold variances (one per fold) or bootstrap reps (one array of boostrap reps per fold)
    if num_boot>0:
        bca_a_numerator, bca_a_denominator = 0, 0
            
    for fold in range(Graph().num_folds):  
        # get initial estimate
        curr_est = forward_pass(order,fold)
        # estimate EIF
        backward_pass(order,fold)

        curr_uncertainty = 0
        if ci_type in ['bca','percentile_boot']:
            curr_bca_a_numerator, curr_bca_a_denominator = 0, 0
        for a in order:
            if isinstance(a,Distribution):
                val_inds = a.value['splits'][fold]['val_inds']
                
                inf_func = a.adjoint(a.value["data"].iloc[val_inds])
                curr_correction = np.mean(inf_func)
                curr_est += curr_correction
                if ci_type in ['bca','percentile_boot']:
                    curr_uncertainty += np.array([np.mean(np.random.choice(inf_func, len(inf_func), replace=True)) - curr_correction for _ in range(num_boot)])
                    if ci_type=='bca':
                        curr_bca_a_numerator += np.sum(inf_func**3)/len(val_inds)**3
                        curr_bca_a_denominator += np.sum(inf_func**2)/len(val_inds)**2
                else:
                    curr_uncertainty += np.var(inf_func)/len(val_inds)
                
        ests += [curr_est]
        uncertainty += [curr_uncertainty]
        
    est = np.mean(ests)
    if ci_type in ['bca','percentile_boot']:
        bootstrap_reps = np.mean(uncertainty, axis=0)
        se = np.std(bootstrap_reps)
        if ci_type=='percentile_boot':
            lower, upper = np.quantile(bootstrap_reps, [(1 - level) / 2, (1 + level) / 2])
        else:
            bca_a = 1/6 * np.sum(curr_bca_a_numerator)/np.sum(curr_bca_a_denominator)**(1.5) # see Eq. 5.28 in Bootstrap Methods and their Application by Davison and Hinkley
            bca_w = norm.ppf(np.mean(bootstrap_reps <= 0), loc=0., scale=1.)
            bca_z_alpha = norm.ppf([(1 - level) / 2, (1 + level) / 2], loc=0., scale=1.)
            lower, upper = np.quantile(bootstrap_reps, norm.cdf(bca_w + (bca_w + bca_z_alpha)/(1-bca_a*(bca_w-bca_z_alpha)), loc=0., scale=1.)) # see Eq. 5.20 in Bootstrap Methods and their Application by Davison and Hinkley
    else:
        var = np.mean(uncertainty)/Graph().num_folds
        se = np.sqrt(var)
        lower, upper = se * norm.ppf([(1 - level) / 2, (1 + level) / 2], loc=0., scale=1.)

    ci = np.array([est+lower,est+upper])

    # clear the computation graph
    # TO DO: possibly make an optional argument for this in future versions, so that nuisance estimates can be reused across different parameters
    Graph().clear()

    return {
        'est': est,
        'se': se,
        'ci': ci}


