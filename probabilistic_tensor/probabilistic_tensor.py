import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps


def zero_superdiag_trailing(x: torch.Tensor, N: int = None, inplace: bool = False, fill: float = 0) -> torch.Tensor:
# def zero_superdiag_trailing(x: torch.Tensor, N: int | None = None, inplace: bool = False, fill: float = -1) -> torch.Tensor:
    """
    Zero entries where the last N indices are all equal.
    If N is None, it is inferred as the count of trailing dims that all have the same size as the last dim.

    Examples
      x shape (1, 3, 3, 3) and N None  ⇒ N becomes 3
      x shape (B, d, d) and N 2         ⇒ zeros along the usual matrix diagonal for every batch item
    """
    if x.dim() == 0:
        return x

    # Infer N if not provided
    if N is None:
        d_last = x.size(-1)
        N = 1
        k = 2
        while k <= x.dim() and x.size(-k) == d_last:
            N += 1
            k += 1

    if N < 1 or N > x.dim():
        raise ValueError(f"N must be between 1 and {x.dim()}, got {N}.")

    # All last N dims must be the same size to define a super diagonal
    d = x.size(-1)
    for k in range(2, N + 1):
        if x.size(-k) != d:
            raise ValueError("The last N dims must all be equal in size to zero the super diagonal.")

    i = torch.arange(d, device=x.device)
    idx = (slice(None),) * (x.dim() - N) + (i,) * N

    if inplace:
        x[idx] = fill
        return x
    else:
        y = x.clone()
        y[idx] = fill
        return y


def zero_any_equal_trailing(x: torch.Tensor, N: int, fill: float) -> torch.Tensor:
    """
    Zero entries where any pair among the last N indices are equal.
    Enforces all different over the last N dims.
    """
    d = x.size(-1)
    # Build a boolean mask over the last N dims
    mask = torch.zeros((d,) * N, dtype=torch.bool, device=x.device)
    for a in range(N):
        for b in range(a + 1, N):
            eye = torch.eye(d, dtype=torch.bool, device=x.device)
            view = [1] * N
            view[a] = d
            view[b] = d
            mask |= eye.view(*view)
    # Expand mask to the full tensor shape and apply
    full_mask = mask.expand(*x.shape[-N:])
    full_mask = full_mask.reshape((1,) * (x.dim() - N) + full_mask.shape)
    full_mask = full_mask.expand(*x.shape)
    return x.masked_fill(full_mask, fill)

forbid_equal: str = "any"
def and_op(*concepts):
    """
    Compose more than two probabilistic concepts using AND.
    
    Each concept is a ProbabilisticTensor that carries:
      - .tensor: its underlying tensor
      - .vars: a list of variable names (e.g., ["x1"], ["x1", "x2"], etc.)
      
    The function first determines the full set of variables used (global_vars)
    and the corresponding domain sizes. Then for each concept, it permutes and
    unsqueezes its tensor so that its dimensions align with the global order.
    Finally, it broadcasts the tensor to the full global shape and computes the
    element-wise minimum (representing AND) across all the concepts.
    """
    concepts = list(concepts)
    if not concepts:
        raise ValueError("No concepts provided for AND operation.")

    # 1. Determine the global order of variables (order of first appearance).
    global_vars = []
    for concept in concepts:
        if isinstance(concept, torch.Tensor):
            continue
        # if len(concept.vars) == 2:
        #     ## set diagonal to zero if 2 vars
        #     concept.tensor = zero_superdiag_trailing(concept.tensor, 2, inplace=True)
        for var in concept.vars:
            if var not in global_vars:
                global_vars.append(var)
    # 2. If any of the concepts is a tensor, convert it to a ProbabilisticTensor and use the vars from the other concepts based on the shape size.    
    for i, concept in enumerate(concepts):
        if isinstance(concept, torch.Tensor):
            concepts[i] = ProbabilisticTensor(concept, vars=global_vars[:len(concept.shape)])
    ## bring all to the same device as the first concept's tensor
    # Determine device from the first tensor
    device = None
    for concept in concepts:
        if isinstance(concept, torch.Tensor):
            device = concept.device
            break
        if hasattr(concept, 'tensor') and concept.tensor is not None:
            device = concept.tensor.device
            break
    
    # If no device found, use CPU as default
    if device is None:
        device = torch.device('cpu')
    
    # Move all concepts to the same device
    for concept in concepts:
        if isinstance(concept, ProbabilisticTensor) and concept.tensor is not None:
            concept.tensor = concept.tensor.to(device)   
    # 3. Determine the domain size for each variable.
    #    We assume that any concept including a given variable uses the same domain size.
    var_sizes = {}
    max_var_size = 0
    total_num_ops = 0
    for concept in concepts:
        num_ops = getattr(concept, "num_ops", 0)
        total_num_ops += num_ops
        for i, var in enumerate(concept.vars):
            if len(concept.tensor.shape) == 0:
                size = 0
            else:
                size = concept.tensor.shape[i]
            if var in var_sizes and var_sizes[var] != size:
                raise ValueError(f"Inconsistent domain size for variable {var}.")
            var_sizes[var] = size
            max_var_size = max(max_var_size, size)
    global_shape = [max_var_size for var in global_vars]

    # 4. Expand each concept's tensor to have shape corresponding to global_vars.
    expanded_tensors = []
    for concept in concepts:
        local_vars = concept.vars  # e.g., ["x2"] or ["x1", "x2"]
        local_tensor = concept.tensor

        # (a) Permute the tensor so that its axes follow the order in global_vars.
        #     For each variable in global_vars that is in the concept, record its index.
        perm_order = [local_vars.index(var) for var in global_vars if var in local_vars]
        if perm_order != list(range(len(local_vars))):
            new_tensor = local_tensor.permute(perm_order)
        else:
            new_tensor = local_tensor

        # (b) Insert missing dimensions. For every variable in global_vars not in local_vars,
        #     unsqueeze the tensor at that axis.
        for i, var in enumerate(global_vars):
            if var not in local_vars:
                new_tensor = new_tensor.unsqueeze(i)
        # (c) Broadcast to the global shape.
        if global_shape:
            # Check if we have extra dimensions that need to be squeezed out
            target_numel = 1
            for s in global_shape: target_numel *= s
            if new_tensor.numel() == target_numel:
                new_tensor = new_tensor.reshape(global_shape)
            elif new_tensor.dim() > len(global_shape):
                new_tensor = new_tensor.squeeze(0)
            new_tensor = new_tensor.expand(*global_shape)
        expanded_tensors.append(new_tensor)
    # expanded_tensors = zero_superdiag
    # 5. Compute elementwise minimum (min = probabilistic AND) over all expanded tensors.
    # print("AND operation with", ProbabilisticTensor.get_and_op(), "tensors.")
    stacked_tensors = torch.stack(expanded_tensors, dim=0)

    # 5. Compute elementwise minimum (min = probabilistic AND) over all expanded tensors.
    # print("AND operation with", ProbabilisticTensor.get_and_op(), "tensors.")
    if ProbabilisticTensor.get_and_op() == "min":
        result_tensor = stacked_tensors.amin(dim=0)
    else: ## And type multiply
        result_tensor = stacked_tensors.prod(dim=0)
    # if result_tensor.shape == (1,):
    #     result_tensor = result_tensor.squeeze(0)
    N = len(global_vars)
    if forbid_equal == "super" and N >= 2:
        result_tensor = zero_superdiag_trailing(result_tensor, N, inplace=False, fill=0.0)  # CHANGED
    elif forbid_equal == "any" and N >= 2:
        result_tensor = zero_any_equal_trailing(result_tensor, N, fill=0.0)  # already non-inplace

    return ProbabilisticTensor(result_tensor, vars=global_vars)

def or_op(*concepts):
    """
    Compose more than two probabilistic concepts using OR.
    
    Each concept is a ProbabilisticTensor that carries:
      - .tensor: its underlying tensor
      - .vars: a list of variable names (e.g., ["x1"], ["x1", "x2"], etc.)
      
    The function first determines the full set of variables used (global_vars)
    and the corresponding domain sizes. Then for each concept, it permutes and
    unsqueezes its tensor so that its dimensions align with the global order.
    Finally, it broadcasts the tensor to the full global shape and computes the
    element-wise maximum (representing OR) across all the concepts.
    """
    concepts = list(concepts)
    if not concepts:
        raise ValueError("No concepts provided for OR operation.")

    # 1. Determine the global order of variables (order of first appearance).
    global_vars = []
    for concept in concepts:
        if isinstance(concept, torch.Tensor):
            continue
        for var in concept.vars:
            if var not in global_vars:
                global_vars.append(var)
    # 2. If any of the concepts is a tensor, convert it to a ProbabilisticTensor and use the vars from the other concepts based on the shape size.    
    for i, concept in enumerate(concepts):
        if isinstance(concept, torch.Tensor):
            concepts[i] = ProbabilisticTensor(concept, vars=global_vars[:len(concept.shape)])

    # 3. Determine the domain size for each variable.
    #    We assume that any concept including a given variable uses the same domain size.
    var_sizes = {}
    for concept in concepts:
        for i, var in enumerate(concept.vars):
            if len(concept.tensor.shape) == 0:
                size = 0
            else:
                size = concept.tensor.shape[i]
            if var in var_sizes and var_sizes[var] != size:
                raise ValueError(f"Inconsistent domain size for variable {var}.")
            var_sizes[var] = size
    global_shape = [var_sizes[var] for var in global_vars]

    # 4. Expand each concept's tensor to have shape corresponding to global_vars.
    expanded_tensors = []
    for concept in concepts:
        local_vars = concept.vars  # e.g., ["x2"] or ["x1", "x2"]
        local_tensor = concept.tensor

        # (a) Permute the tensor so that its axes follow the order in global_vars.
        #     For each variable in global_vars that is in the concept, record its index.
        perm_order = [local_vars.index(var) for var in global_vars if var in local_vars]
        if perm_order != list(range(len(local_vars))):
            new_tensor = local_tensor.permute(perm_order)
        else:
            new_tensor = local_tensor

        # (b) Insert missing dimensions. For every variable in global_vars not in local_vars,
        #     unsqueeze the tensor at that axis.
        for i, var in enumerate(global_vars):
            if var not in local_vars:
                new_tensor = new_tensor.unsqueeze(i)
        # (c) Broadcast to the global shape.
        if global_shape:
            new_tensor = new_tensor.expand(*global_shape)
        expanded_tensors.append(new_tensor)
    # 5. Compute elementwise maximum (max = probabilistic OR) over all expanded tensors.
    result_tensor = torch.stack(expanded_tensors, dim=0).amax(dim=0)
    return ProbabilisticTensor(result_tensor, vars=global_vars)    

def serializable(data):
    if data is None:
        return None
    if data is True or data is False:
        return data
    if isinstance(data, (int, float, str)):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    if isinstance(data, list):
        return [serializable(d) for d in data]
    if isinstance(data, dict):
        return {k: serializable(v) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(serializable(d) for d in data)
    if isinstance(data, set):
        return {serializable(d) for d in data}
    if isinstance(data, ProbabilisticTensor):
        return data.tensor.detach().cpu().to(torch.float32).numpy().tolist()
    if data is Ellipsis:
        return "..."
    if isinstance(data, slice):
        return {"start": getattr(data,"start", None), "stop": getattr(data,"stop", None), "step": getattr(data,"step", None), "indices": getattr(data,"indices", None)}
    elif callable(data):  # Handles cases where the indices method is mistakenly included
        return str(data)

    return data
class ProbabilisticTensor:
    training = False
    cache = []

    @classmethod
    def set_and_op(cls, op):
        if op not in ["min", "mul"]:
            raise ValueError("Invalid AND operation. Choose 'min' or 'mul'.")
        cls.and_op = op
    @classmethod
    def get_and_op(cls):
        return cls.and_op

    and_op = "mul"  # or "multiply"
    
    @classmethod
    def start_cache(cls):
        cls.cache = []
    
    @classmethod
    def end_cache(cls):
        temp_cache = cls.cache[:]
        cls.cache = []
        return temp_cache
    

    def __init__(self, tensor, vars=None, normalized=False, extra_info=None):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        self.tensor = tensor
        self.normalized = normalized
        self.vars = vars if vars is not None else [f"x{i}" for i in range(1, len(tensor.shape)+1)]
        if extra_info is not None:
            inputs = {
                "args": (extra_info),
                "kwargs": {},
                "self.tensor": serializable(self),
            }
            # Cache the action
            self.__class__._cache_action("__init__", inputs, None)

    def __call__(self, *args):
        return ProbabilisticTensor(self.tensor, vars=args)

    def __getattr__(self, name):
        """
        Fallback to torch.Tensor methods if not implemented in ProbabilisticTensor.

        This allows calling any torch.Tensor method on a ProbabilisticTensor.
        If the result is a tensor, it's wrapped back into a ProbabilisticTensor.
        """
        # Avoid infinite recursion for attributes accessed during __init__
        if name in ('tensor', 'vars', 'normalized', 'cache', 'training', 'and_op'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Get the attribute from the underlying tensor
        tensor_attr = getattr(self.tensor, name)

        # If it's a method, wrap it to return ProbabilisticTensor when appropriate
        if callable(tensor_attr):
            def wrapped_method(*args, **kwargs):
                result = tensor_attr(*args, **kwargs)
                # If result is a tensor, wrap it in ProbabilisticTensor
                if isinstance(result, torch.Tensor):
                    # Try to preserve vars if shape is compatible
                    if result.shape == self.tensor.shape:
                        return ProbabilisticTensor(result, vars=self.vars)
                    else:
                        return ProbabilisticTensor(result)
                return result
            return wrapped_method

        # If it's a property/attribute, return it directly
        return tensor_attr

    @classmethod
    def _cache_action(cls, action_name, inputs, result):
        """Helper method to cache an action."""
        ## result.tensor.detach().cpu().numpy().tolist() if isinstance(result, ProbabilisticTensor) else result
        cls.cache.append({
            "action": action_name,
            "inputs": serializable(inputs),
            "result": serializable(result) ,
        })
    @staticmethod
    def cache_action(method):
        """Decorator to cache method calls."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            action_name = method.__name__
            # Prepare inputs for caching
            inputs = {
                "args": [serializable(arg) for arg in args],
                "kwargs": serializable(kwargs),
                "self.tensor": serializable(self),
            }
            ## serialize the inputs
            # Cache the action
            self.__class__._cache_action(action_name, inputs, result)
            return result
        return wrapper


    def __repr__(self):
        return str(self.tensor)
    
    def __str__(self):
        return str(self.tensor)

    @cache_action.__get__(object)
    def __and__(self, other):
        return and_op(self, other)
    
    @cache_action.__get__(object)
    def normalize_tensor(self, tensor, dim=-1):
        # Perform linear normalization with zero as min
        min_vals = tensor.min(dim=dim, keepdim=True).values
        max_vals = tensor.max(dim=dim, keepdim=True).values

        return (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add epsilon to avoid division by zero

    @cache_action.__get__(object)
    def __rand__(self, other):
        return ProbabilisticTensor(self.__and__(other))
    ## OR

    @cache_action.__get__(object)
    def __or__(self, other):
        return or_op(self, other)
    
    ## not
    @cache_action.__get__(object)
    def __invert__(self):
        device = self.tensor.device
        if len(self.tensor.shape) == 1:
            invert = torch.ones_like(self.tensor, device=device)
        else:
            invert = torch.ones_like(self.tensor, device=device) - torch.eye(self.tensor.shape[0], device=device)
        return ProbabilisticTensor(invert - self.tensor, vars=self.vars)
    @cache_action.__get__(object)
    def __gt__(self, other):
        _count_margin = 0.25
        _count_tau = 0.25
        if isinstance(other, ProbabilisticTensor):
            other_value = other.tensor
        else:
            other_value = other
        if self.training:
            rv = torch.sigmoid((self.tensor - other_value +  _count_margin) / _count_tau)
        else:
            rv =  (self.tensor > other_value).float()
        return ProbabilisticTensor(rv)
    @cache_action.__get__(object)
    def __lt__(self, other):
        _count_margin = 0.25
        _count_tau = 0.25
        if isinstance(other, ProbabilisticTensor):
            other_value = other.tensor
        else:
            other_value = other
        if self.training:
            # Differentiable: sigmoid of negative difference
            rv = torch.sigmoid((other_value - self.tensor + _count_margin) / _count_tau)
        else:
            rv = (self.tensor < other_value).float()
        return ProbabilisticTensor(rv)
    
    @cache_action.__get__(object)
    def __eq__(self, other):
        _count_margin = 0.25
        _count_tau = 0.25
        if isinstance(other, ProbabilisticTensor):
            other_value = other.tensor
        else:
            other_value = other
        rv = torch.sigmoid((2 * _count_margin - (self.tensor - other_value).abs()) / (2 * _count_margin) / _count_tau)
        return ProbabilisticTensor(rv)
    @cache_action.__get__(object)
    def __ne__(self, other):
        _count_margin = 0.25
        _count_tau = 0.25
        if isinstance(other, ProbabilisticTensor):
            other_value = other.tensor
        else:
            other_value = other
        if self.training:
            # Differentiable: 1 - eq (complement of equality)
            eq_score = torch.sigmoid((2 * _count_margin - (self.tensor - other_value).abs()) / (2 * _count_margin) / _count_tau)
            rv = 1 - eq_score
        else:
            # Probabilistic: check if absolute difference > margin
            rv = ((self.tensor - other_value).abs() > _count_margin).float()
        return ProbabilisticTensor(rv)
    @cache_action.__get__(object)
    def __ge__(self, other):
        return self.__gt__(other) | self.__eq__(other)
    @cache_action.__get__(object)
    def __le__(self, other):
        return self.__lt__(other) | self.__eq__(other)
    @cache_action.__get__(object)
    def __add__(self, other):
        if isinstance(other, ProbabilisticTensor):
            return ProbabilisticTensor(self.tensor + other.tensor, vars=self.vars)
        return ProbabilisticTensor(self.tensor + other, vars=self.vars)
    @cache_action.__get__(object)
    def __sub__(self, other):
        return ProbabilisticTensor(self.tensor - other.tensor, vars=self.vars)
    @cache_action.__get__(object)
    def __mul__(self, other):
        if isinstance(other, ProbabilisticTensor):
            return ProbabilisticTensor(self.tensor * other.tensor, vars=self.vars)
        return ProbabilisticTensor(self.tensor * other, vars=self.vars)
    @cache_action.__get__(object)
    def __truediv__(self, other):
        if isinstance(other, ProbabilisticTensor):
            return ProbabilisticTensor(self.tensor / other.tensor)
        return ProbabilisticTensor(self.tensor / other)
    @cache_action.__get__(object)
    def __floordiv__(self, other):
        if isinstance(other, ProbabilisticTensor):
            return ProbabilisticTensor(self.tensor // other.tensor, vars=self.vars)
        return ProbabilisticTensor(self.tensor // other, vars=self.vars)
    @cache_action.__get__(object)
    def __mod__(self, other):
        if isinstance(other, ProbabilisticTensor):
            return ProbabilisticTensor(self.tensor % other.tensor)
        return ProbabilisticTensor(self.tensor % other)
    @cache_action.__get__(object)
    def __pow__(self, other):
        if isinstance(other, ProbabilisticTensor):
            return ProbabilisticTensor(self.tensor ** other.tensor)
        return ProbabilisticTensor(self.tensor ** other)
    @cache_action.__get__(object)
    def __abs__(self):
        return ProbabilisticTensor(self.tensor.abs(), vars=self.vars)
    @cache_action.__get__(object)
    def __neg__(self):
        return ProbabilisticTensor(-self.tensor, vars=self.vars)
    @cache_action.__get__(object)
    def __pos__(self):
        return self

    @cache_action.__get__(object)
    def exists(self, *vars, threshold=0.5, temperature=0.1, **kwargs):
        """
        Existential quantifier: checks if any object satisfies the predicate.

        Args:
            *vars: Variables to eliminate. If empty, returns scalar existence.
            threshold: The decision threshold (default 0.5).
            temperature: Controls sharpness of the soft threshold (smaller = sharper).
                        Default 0.1 gives a reasonably sharp transition.
            **kwargs: Legacy support for 'var' keyword argument.

        Returns:
            ProbabilisticTensor with soft existence score:
            - Close to 1 when max value is above threshold
            - Close to 0 when max value is below threshold
            - Differentiable for gradient-based training
        """
        # Handle legacy 'var' keyword argument
        if 'var' in kwargs:
            if vars:
                raise ValueError("Cannot specify both positional arguments and 'var' keyword argument.")
            if kwargs['var'] is not None:
                vars = (kwargs['var'],)

        def soft_existence(tensor):
            """
            Soft existence using sigmoid around threshold.
            sigmoid((max_val - threshold) / temperature)

            This is differentiable and:
            - Returns ~1 when max >> threshold
            - Returns ~0 when max << threshold
            - Returns 0.5 when max == threshold
            """
            max_val = tensor.amax() if tensor.dim() > 0 else tensor
            return torch.sigmoid((max_val - threshold) / temperature)

        # If vars is empty (no positional args and var=None or not provided), perform scalar existence
        if not vars:
            soft_score = soft_existence(self.tensor)
            return ProbabilisticTensor(soft_score, vars=[])

        # Projected existence over multiple variables
        dims_to_reduce = []
        for v in vars:
            if v in self.vars:
                dims_to_reduce.append(self.vars.index(v))
            else:
                raise ValueError(f"Variable {v} not found in tensor variables {self.vars}")

        # Remove duplicates and sort dimensions
        dims_to_reduce = tuple(sorted(set(dims_to_reduce), reverse=True))

        # First reduce to get max along specified dimensions
        reduced_tensor = self.tensor.amax(dim=dims_to_reduce)

        # Apply soft existence to each remaining element
        soft_score = torch.sigmoid((reduced_tensor - threshold) / temperature)

        new_vars = [v for i, v in enumerate(self.vars) if i not in dims_to_reduce]
        return ProbabilisticTensor(soft_score, vars=new_vars)

    
    @cache_action.__get__(object)
    def count(self, *vars, threshold=0.5, temperature=0.1):
        """
        Count objects satisfying the predicate.

        Args:
            *vars: Variables to group by. If empty, returns scalar count.
            threshold: Decision threshold (default 0.5).
            temperature: Sharpness of soft threshold in training mode.

        Returns:
            Training mode: Differentiable soft count using sigmoid.
            Inference mode: Hard count of values >= threshold.
        """
        if self.training:
            # Differentiable: soft count using sigmoid
            soft_indicators = torch.sigmoid((self.tensor - threshold) / temperature)
        else:
            # Hard threshold for inference
            soft_indicators = (self.tensor >= threshold).float()

        if not vars:
            
            return ProbabilisticTensor(soft_indicators.sum())

        # Sum over variables NOT in vars (keep the specified vars)
        vars_list = list(vars)
        result = soft_indicators
        # Need to reduce dimensions from high to low to avoid index shifting
        dims_to_reduce = []
        for i, var in enumerate(self.vars):
            if var not in vars_list:
                dims_to_reduce.append(i)

        if dims_to_reduce:
            result = result.sum(dim=tuple(dims_to_reduce))

        return ProbabilisticTensor(result)
    def __round__(self, decimals=0):
        return ProbabilisticTensor(self.tensor.round(decimals), vars=self.vars)
    @cache_action.__get__(object)
    def argmax(self, var=None, k=None, dim=None, object_id=None, temperature=0.1, return_soft=False):
        """
        Get the index of the maximum value.

        Args:
            var: Variable name to get argmax for.
            k: Return k-th largest index.
            dim: Dimension to reduce.
            object_id: Get argmax for specific object.
            temperature: Temperature for soft argmax in training mode.
            return_soft: If True, return soft distribution instead of index.

        Returns:
            Training mode (return_soft=True): Differentiable soft distribution (softmax).
            Inference mode: Hard integer index.
        """
        if self.training and return_soft:
            # Return differentiable soft argmax (attention weights)
            if dim is None:
                flat = self.tensor.flatten()
                soft_idx = F.softmax(flat / temperature, dim=0)
                return ProbabilisticTensor(soft_idx.view(self.tensor.shape), vars=self.vars)
            else:
                soft_idx = F.softmax(self.tensor / temperature, dim=dim)
                return ProbabilisticTensor(soft_idx, vars=self.vars)

        # Hard argmax for inference
        if var is not None:
            if isinstance(var, str):
                coords = torch.unravel_index(self.tensor.argmax(), self.tensor.shape)
                var_index = self.vars.index(var)
                return coords[var_index].item()
            else:
                raise ValueError("var must be a valid variable.")

        if object_id is not None:
            return int(self.tensor[object_id].argmax(dim=dim).detach().cpu().numpy().item())

        if k is not None:
            return int(self.tensor.topk(k)[1][k-1])

        if len(self.tensor.shape) > 1 and self.tensor.shape[1] == self.tensor.shape[0]:
            return int(self.tensor.sum(dim=1).argmax(dim=dim).detach().cpu().numpy().item())

        return int(self.tensor.argmax(dim=dim).detach().cpu().numpy().item())

    @cache_action.__get__(object)
    def topk(self, var=None,  k=1, dim=-1):
        """
        Returns the top k indices along the specified dimension.
        """
        # return self.tensor.topk(k=k, dim=dim)[1]
        if isinstance(var, int):
            k = var
            var = None
        if var is not None:
            var_index = self.vars.index(var)
            return self.tensor.topk(k=k, dim=var_index)[1]
        else:
            return self.tensor.topk(k=k, dim=dim)[1]   
             
    @cache_action.__get__(object)
    def argmin(self, var=None, k=None, dim=None, object_id=None, temperature=0.1, return_soft=False):
        """
        Get the index of the minimum value.

        Args:
            var: Variable name to get argmin for.
            k: Return k-th smallest index.
            dim: Dimension to reduce.
            object_id: Get argmin for specific object.
            temperature: Temperature for soft argmin in training mode.
            return_soft: If True, return soft distribution instead of index.

        Returns:
            Training mode (return_soft=True): Differentiable soft distribution (softmax on negated values).
            Inference mode: Hard integer index.
        """
        if self.training and return_soft:
            # Return differentiable soft argmin (softmax on negated tensor)
            if dim is None:
                flat = self.tensor.flatten()
                soft_idx = F.softmax(-flat / temperature, dim=0)
                return ProbabilisticTensor(soft_idx.view(self.tensor.shape), vars=self.vars)
            else:
                soft_idx = F.softmax(-self.tensor / temperature, dim=dim)
                return ProbabilisticTensor(soft_idx, vars=self.vars)

        # Hard argmin for inference
        if var is not None:
            if isinstance(var, str):
                coords = torch.unravel_index(self.tensor.argmin(), self.tensor.shape)
                var_index = self.vars.index(var)
                return coords[var_index].item()
            else:
                raise ValueError("var must be a valid variable.")

        if object_id is not None:
            return int(self.tensor[object_id].argmin(dim=dim).detach().cpu().numpy().item())

        if k is not None:
            return int(self.tensor.topk(k, largest=False)[1][k-1])
    
    @cache_action.__get__(object)
    def argsort(self, var=None, dim=-1):
        """
        Returns the indices that would sort the tensor along the specified dimension.
        """
        if var is not None:
            var_index = self.vars.index(var)
            return self.tensor.argsort(dim=var_index)
        else:
            return self.tensor.argsort(dim=dim)
    
    @cache_action.__get__(object)
    def max(self, var=None, dim=-1):
        return ProbabilisticTensor(self.tensor.max(dim=dim).values)
    # @cache_action.__get__(object)
    def amax(self, dim=-1):
        return ProbabilisticTensor(self.tensor.amax(dim=dim).values)
    @cache_action.__get__(object)
    def min(self, dim=-1):
        return ProbabilisticTensor(self.tensor.min(dim=dim).values)
    @cache_action.__get__(object)
    def amin(self, dim=-1):
        return ProbabilisticTensor(self.tensor.amin(dim=dim).values)
    @cache_action.__get__(object)
    def sum(self, dim=-1):
        return ProbabilisticTensor(self.tensor.sum(dim=dim))
    @cache_action.__get__(object)
    def __bool__(self):
        return (self.tensor >= 0.5).all().item()
    @cache_action.__get__(object)
    def __len__(self):
        return len(self.tensor)
    @cache_action.__get__(object)
    def __getitem__(self, key):
        if isinstance(key, str) and key in self.vars:
            key = self.vars.index(key)
        elif isinstance(key, tuple):
            key = tuple([int(k.cpu().numpy().item()) if isinstance(k, torch.Tensor) else k for k in key])
        elif isinstance(key, torch.Tensor):
            key = int(key.cpu().numpy().item())
        return ProbabilisticTensor(self.tensor.__getitem__(key))
    @cache_action.__get__(object)
    def __setitem__(self, key, value):
        self.tensor[key] = value.tensor

    @cache_action.__get__(object)
    def normalize(self, dim=-1):
        ## perform linear normalization with zero as min
        return ProbabilisticTensor((self.tensor - self.tensor.min(dim=dim).values) / (self.tensor.max(dim=dim).values))
    @cache_action.__get__(object)
    def __sub__(self, other):
        if isinstance(other, ProbabilisticTensor):
            other = other.tensor
        return ProbabilisticTensor(self.tensor - other)
    
    @cache_action.__get__(object)
    def __rsub__(self, other):
        return ProbabilisticTensor(other - self.tensor)
    
    @cache_action.__get__(object)
    def any(self, threshold=0.5, temperature=0.1, **kwargs):
        """
        Check if any element satisfies the predicate along a dimension.

        Args:
            threshold: Decision threshold (default 0.5).
            temperature: Sharpness in training mode.
            dim/axis: Dimension to reduce (default -1).

        Returns:
            Training mode: Differentiable soft any using sigmoid(max - threshold).
            Inference mode: Hard check if max >= threshold.
        """
        dim = kwargs.get("dim", kwargs.get("axis", -1))

        if self.training:
            # Differentiable: sigmoid of (max - threshold)
            max_val = self.tensor.amax(dim=dim)
            rv = torch.sigmoid((max_val - threshold) / temperature)
        else:
            # Probabilistic: check if max value >= threshold
            max_val = self.tensor.amax(dim=dim)
            rv = (max_val >= threshold).float()

        return ProbabilisticTensor(rv)
    
    @cache_action.__get__(object)
    def all(self, threshold=0.5, temperature=0.1, dim=-1):
        """
        Check if all elements satisfy the predicate along a dimension.

        Args:
            threshold: Decision threshold (default 0.5).
            temperature: Sharpness in training mode.
            dim: Dimension to reduce (default -1).

        Returns:
            Training mode: Differentiable soft all using sigmoid(min - threshold).
            Inference mode: Hard boolean all.
        """
        tensor = self.tensor
        if len(self.tensor.shape) > 1 and self.tensor.shape[1] == self.tensor.shape[0]:
            tensor = tensor + torch.eye(tensor.shape[0], device=tensor.device)

        if self.training:
            # Differentiable: sigmoid of (min - threshold)
            min_val = tensor.amin(dim=dim)
            rv = torch.sigmoid((min_val - threshold) / temperature)
        else:
            rv = (tensor >= threshold).all(dim=dim).float()

        return ProbabilisticTensor(rv)
    @cache_action.__get__(object)
    def forall(self, var=None, threshold=0.5, temperature=0.1, dim=-1):
        """
        Universal quantifier over a specific variable.

        Args:
            var: Variable to quantify over. If None, reduces over dim.
            threshold: Decision threshold (default 0.5).
            temperature: Sharpness in training mode.
            dim: Dimension to reduce if var is None.

        Returns:
            Training mode: Differentiable soft forall using sigmoid(min - threshold).
            Inference mode: Hard boolean all.
        """
        tensor = self.tensor

        # Add eye for diagonal elements (self-relations are true)
        if len(tensor.shape) > 1 and tensor.shape[1] == tensor.shape[0]:
            tensor = tensor + torch.eye(tensor.shape[0], device=tensor.device)

        if var is None:
            if self.training:
                min_val = tensor.amin(dim=dim)
                rv = torch.sigmoid((min_val - threshold) / temperature)
            else:
                rv = (tensor >= threshold).all(dim=dim).float()
            return ProbabilisticTensor(rv)

        var_index = self.vars.index(var)

        if self.training:
            # Differentiable: sigmoid of (min - threshold) along the variable dimension
            min_val = tensor.amin(dim=var_index)
            reduced_tensor = torch.sigmoid((min_val - threshold) / temperature)
        else:
            mask = tensor >= threshold
            reduced_tensor = mask.all(dim=var_index).float()

        new_vars = self.vars[:var_index] + self.vars[var_index+1:]
        return ProbabilisticTensor(reduced_tensor, vars=new_vars)
    
    @cache_action.__get__(object)
    def mask(self, other):
        """
        Mask the current tensor with another tensor.
        This is here for backward compatibility.
        """
        if self.tensor.shape == other.tensor.shape:
            return ProbabilisticTensor(torch.stack([self.tensor,other.tensor], dim=-1).prod(dim=-1), vars=self.vars)
        else:
            if len(self.tensor.shape) == 1:
                a = self.normalize().tensor
                b = other.tensor
            else:
                a = other.normalize().tensor
                b = self.tensor
            for _ in range(len(b.shape)):
                b = a.unsqueeze(1).repeat((1,a.shape[0])) * b
                b = b.transpose(0,1)
            return ProbabilisticTensor(b, vars=self.vars)

    @cache_action.__get__(object)
    def __rshift__(self, other):
        """
        Logical IMPLIES: A >> B ≡ ~A | B
        """
        return self.implies(other)
    
    @cache_action.__get__(object)
    def implies(self, other, logic="kleene"):
        if logic == "lukasiewicz":
            return ProbabilisticTensor(
                torch.clamp(1 - self.tensor + other.tensor, max=1.0), 
                vars=list(set(self.vars + other.vars))
            )
        else:  # if logic == "kleene":
            return ~self | other

    @cache_action.__get__(object)
    def not_implies(self, other):
        """
        Negated Implication: A & ~B
        """
        return self & ~other

    @cache_action.__get__(object)
    def __xor__(self, other):
        """
        Logical XOR: (A | B) & ~(A & B)
        """
        return (self | other) & ~(self & other)

    @cache_action.__get__(object)
    def __rshift__(self, other):
        """
        Logical IMPLIES: A >> B ≡ ~A | B
        """
        return ~self | other

    @cache_action.__get__(object)
    def __lshift__(self, other):
        """
        NOT IMPLIES: A << B ≡ A & ~B
        """
        return self & ~other
    
    @cache_action.__get__(object)
    def iota(self, var=None, method='linear'):
        """
        IOTA operator: Aggregates over other variables and normalizes the result
        to create a distribution or selection score for the target variable.

        Args:
            var (str): The variable to create a distribution for. Defaults to the first variable.
            method (str): 'softmax' or 'linear' normalization.

        Returns:
            ProbabilisticTensor: Normalized 1D tensor representing the distribution
                                 over the specified variable. The `vars` will contain only `var`.
        """
        if not self.vars:
             if self.tensor.numel() == 1:
                 # Cannot normalize a scalar meaningfully without context. Return as is?
                 # Or assume it's already a probability? Let's return it.
                 # If method is softmax, a single value becomes 1.0.
                 if method == 'softmax':
                     return ProbabilisticTensor(self.tensor / self.tensor , vars=self.vars, normalized=True)
                 else: # Linear normalization of scalar is ambiguous, return original
                     return self
             else:
                 raise ValueError("Cannot perform iota on tensor with dimensions but no variable names.")

        if var is None:
            var = self.vars[0] # Default to the first variable

        if var not in self.vars:
            raise ValueError(f"Variable '{var}' not in tensor vars: {self.vars}")

        var_index = self.vars.index(var)
        num_dims = self.tensor.dim()

        # --- Aggregation Step ---
        dims_to_aggregate = [i for i in range(num_dims) if i != var_index]

        if dims_to_aggregate:
            # Aggregate using maximum possibility across other dimensions
            aggregated_tensor = self.tensor.amax(dim=tuple(dims_to_aggregate))
        else:
            # Tensor is already 1D (or potentially 0D if original was 0D)
            aggregated_tensor = self.tensor

        # Ensure tensor is at least 1D for normalization if it became scalar after aggregation
        if aggregated_tensor.dim() == 0:
            aggregated_tensor = aggregated_tensor.unsqueeze(0) # Make it 1D

        # --- Normalization Step ---
        if aggregated_tensor.numel() == 0:
             # Handle empty tensor case
             normalized_tensor = torch.empty(0, dtype=aggregated_tensor.dtype, device=aggregated_tensor.device)
        elif method == 'softmax':
             # Softmax requires at least 1D, which aggregated_tensor should be now.
             normalized_tensor = F.softmax(aggregated_tensor, dim=0) # Normalize the single dimension
        elif method == 'linear':
            # Linear normalization (scale to [0, 1])
            min_val = aggregated_tensor.min()
            max_val = aggregated_tensor.max()
            range_val = max_val - min_val
            if range_val.item() < 1e-8:
                # All values are the same (or very close)
                # Option 1: Uniform distribution if max_val > 0
                # Option 2: Return the constant value scaled (if possible) or just zeros/ones
                # Let's return uniform distribution 1/N if max_val > 0, else zeros.
                num_elements = aggregated_tensor.numel()
                if max_val.item() > 1e-8 and num_elements > 0 :
                    normalized_tensor = torch.ones_like(aggregated_tensor) / num_elements
                else:
                    normalized_tensor = torch.zeros_like(aggregated_tensor)
            else:
                normalized_tensor = (aggregated_tensor - min_val) / range_val
        else:
             return ProbabilisticTensor(aggregated_tensor, vars=[var], normalized=False)
             raise ValueError(f"Unknown normalization method: {method}. Use 'softmax' or 'linear'.")

        # Return the result with only the target variable
        return ProbabilisticTensor(normalized_tensor, vars=[var], normalized=True)
    
    @cache_action.__get__(object)
    def mean(self, var=None, dim=-1):
        if var is not None:
            var_index = self.vars.index(var)
            return ProbabilisticTensor(self.tensor.mean(dim=var_index))
        return ProbabilisticTensor(self.tensor.mean(dim=dim))


import torch

import matplotlib.pyplot as plt
from PIL import Image

# image = Image.open("images/image.png")
# plt.imshow(image)
# plt.axis("off")
# plt.show()

# One-hot encoded tensors for each concept (size 9)
ProbabilisticTensor.start_cache()
# Colors
yellow = torch.tensor([1, 0, 1, 0])  # Yellow for Object 1 and Object 3
yellow = ProbabilisticTensor(yellow)

blue = torch.tensor([0, 1, 0, 1])    # Blue for Object 2 and Object 4
blue = ProbabilisticTensor(blue)

# Shapes
circle = torch.tensor([1, 0, 0, 0])   # Circle for Object 1
circle = ProbabilisticTensor(circle)

cylinder = torch.tensor([0, 0, 1, 0])  # Cylinder for Object 3
cylinder = ProbabilisticTensor(cylinder)

square = torch.tensor([0, 1, 0, 1])    # Square for Object 2 and Object 4
square = ProbabilisticTensor(square)

# Asymmetrical relationship tensors (4x4)

right_of = torch.tensor([
    [0, 0, 1, 1],  # Object 1 is not right of any object
    [1, 0, 1, 1],  # Object 2 is right of Object 1
    [0, 0, 0, 0],  # Object 3 is right of Object 4
    [0, 0, 0, 0]   # Object 4 is not right of any object
])
right_of = ProbabilisticTensor(right_of)

left_of = torch.tensor([
    [0, 1, 0, 0],  # Object 1 is not left of any object
    [0, 0, 0, 0],  # Object 2 is not left of any object
    [1, 1, 0, 0],  # Object 3 is left of Object 1 and Object 2
    [1, 1, 0, 0]   # Object 4 is left of Object 1 and Object 2
])
left_of = ProbabilisticTensor(left_of)

# Symmetrical relationship tensors (4x4), ordered by position

# Same Row
same_row = torch.tensor([
    [0, 1, 0, 0],  # Object 1 is in the same row as Object 2
    [1, 0, 0, 0],  # Object 2 is in the same row as Object 1
    [0, 0, 0, 0],  # Object 3 has no row partner
    [0, 0, 0, 0]   # Object 4 has no row partner
])
same_row = ProbabilisticTensor(same_row)

# Same Column
same_col = torch.tensor([
    [0, 0, 0, 0],  # Object 1 has no column partner
    [0, 0, 0, 0],  # Object 2 has no column partner
    [0, 0, 0, 1],  # Object 3 is in the same column as Object 4
    [0, 0, 1, 0]   # Object 4 is in the same column as Object 3
])
same_col = ProbabilisticTensor(same_col)

# Same Color
same_color = torch.tensor([
    [0, 0, 1, 0],  # Object 1 shares color with Object 3 (both yellow)
    [0, 0, 0, 1],  # Object 2 shares color with Object 4 (both blue)
    [1, 0, 0, 0],  # Object 3 shares color with Object 1 (both yellow)
    [0, 1, 0, 0],  # Object 4 shares color with Object 2 (both blue)
])
same_color = ProbabilisticTensor(same_color)

same_size = torch.tensor([
    [0, 0, 1, 0],  
    [0, 0, 0, 0],  
    [1, 0, 0, 0],  
    [0, 0, 0, 0],  
])
same_size = ProbabilisticTensor(same_size)

# Same Shape
same_shape = torch.tensor([
    [0, 0, 0, 0],  # Object 1 has no shape partner
    [0, 0, 0, 1],  # Object 2 shares shape with Object 4 (both squares)
    [0, 0, 0, 0],  # Object 3 has no shape partner
    [0, 1, 0, 0],  # Object 4 shares shape with Object 2 (both squares)
])
same_shape = ProbabilisticTensor(same_shape)
### blue and square

blue_square = blue & square
# print("blue_square", blue_square)

### yellow and cylinder
yellow_cylinder = yellow & cylinder
# print("yellow_cylinder", yellow_cylinder)

### blue square and same row as a yellow circle
blue_square = blue & square

yellow_circle = yellow & circle

# print("same_row", same_row)


# print("blue square", blue_square)
# print("left_of", left_of)
# print("yellow_cylinder", yellow_cylinder)

# print('left_of("x2", "x1") & blue_square("x1")', left_of("x2", "x1") & blue_square("x1"))
# print('left_of("x2", "x1") & yellow_cylinder("x2")', left_of("x2", "x1") & yellow_cylinder("x2"))

# print('(left_of("x2", "x1") & yellow_cylinder("x2")).exists()', (left_of("x2", "x1") & yellow_cylinder("x2")).exists())
# print('(left_of("x2", "x1") & yellow_cylinder("x2")).count()', (left_of("x2", "x1") & yellow_cylinder("x2")).count())
# print('(left_of("x2", "x1") & yellow_cylinder("x2")).count("x1")', (left_of("x2", "x1") & yellow_cylinder("x2")).count("x1"))
# print('(left_of("x2", "x1") & yellow_cylinder("x2")).count("x2")', (left_of("x2", "x1") & yellow_cylinder("x2")).count("x2"))
# print('(left_of("x2", "x1") & yellow_cylinder("x2")).count("x1","x2")', (left_of("x2", "x1") & yellow_cylinder("x2")).count("x1","x2"))


# print('(left_of("x2", "x1") & yellow_cylinder("x1")).exists()', (left_of("x2", "x1") & yellow_cylinder("x1")).exists())
# print('(left_of("x2", "x1") & yellow_cylinder("x1")).count()', (left_of("x2", "x1") & yellow_cylinder("x1")).count())
# print('(left_of("x2", "x1") & yellow_cylinder("x1")).count("x1")', (left_of("x2", "x1") & yellow_cylinder("x1")).count("x1"))
# print('(left_of("x2", "x1") & yellow_cylinder("x1")).count("x2")', (left_of("x2", "x1") & yellow_cylinder("x1")).count("x2"))
# print('(left_of("x2", "x1") & yellow_cylinder("x1")).count("x1","x2")', (left_of("x2", "x1") & yellow_cylinder("x1")).count("x1","x2"))

# print('(same_color("x1", "x2").count()', same_color("x1", "x2").count())
# print('(same_color("x1", "x2").count("x1")', same_color("x1", "x2").count("x1"))
# print('(same_color("x1", "x2").count("x2")', same_color("x1", "x2").count("x2"))
# print('(same_color("x1", "x2").count("x1","x2")', same_color("x1", "x2").count("x1","x2"))

# print('(same_color("x1", "x2") & cylinder("x1")).count()', (same_color("x1", "x2") & cylinder("x1")).count())
# print('(same_color("x1", "x2") & cylinder("x1")).count("x1")', (same_color("x1", "x2") & cylinder("x1")).count("x1"))
# print('(same_color("x1", "x2") & cylinder("x1")).count("x2")', (same_color("x1", "x2") & cylinder("x1")).count("x2"))
# print('(same_color("x1", "x2") & cylinder("x1")).count("x1","x2")', (same_color("x1", "x2") & cylinder("x1")).count("x1","x2"))

# print('yellow_cylinder("x1").count()', yellow_cylinder("x1").count())
# print('yellow_cylinder("x1").count("x1")', yellow_cylinder("x1").count("x1"))




# print(and_op(left_of("x1", "x2"), blue_square("x1"), yellow_cylinder("x2")).tensor.shape)
# print((left_of("x1", "x2") & blue_square("x1") & yellow_cylinder("x2")).tensor.shape)
# print(and_op(left_of("x1", "x2"), blue_square("x2"), yellow_cylinder("x1"), yellow_circle("x3")))

# blue.forall("x1")
