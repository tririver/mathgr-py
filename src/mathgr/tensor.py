from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, count, permutations, product
from math import factorial
from typing import Any

import sympy as sp


LatinIdx = tuple([chr(i) for i in range(97, 122)] + [f"y{i}" for i in range(26)])
GreekIdx = tuple([chr(i) for i in range(945, 970)] + [f"omega{i}" for i in range(26)])
LatinCapitalIdx = tuple([chr(i) for i in range(65, 90)] + [f"Y{i}" for i in range(26)])


_INDEX_TYPES: dict[str, "IndexType"] = {}
DefaultDim = sp.Symbol("DefaultDim")
_CONSTANTS: set[sp.Expr] = {DefaultDim}
_METRICS: dict[str, Any] = {}
_METRIC_HEADS: set[sp.Symbol] = set()
_METRIC_INDEX_PAIRS: dict[sp.Symbol, list[tuple["IndexType", "IndexType"]]] = {}
_SYMMETRIES: dict[tuple[sp.Symbol, tuple], list["TensorSymmetry"]] = {}
_UNIQ_COUNTER = count(1)
IdxList: list["IndexType"] = []
IdxUpList: list["IndexType"] = []
IdxDnList: list["IndexType"] = []
SimpHook = []
SimpInto1 = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)
SimpSelect = lambda terms: terms
_MAX_HOOK_ITERATIONS = 10
_MAX_PRODUCT_SYMMETRY_VARIANTS = 256
_MAX_DUMMY_RENAME_VARIANTS = 2048
_MAX_FULL_DUMMY_RENAME_KEYS = 5


class Index(sp.Expr):
    is_commutative = True

    def __new__(cls, head_name: str, label: Any):
        label_expr = _index_label_expr(label)
        return sp.Expr.__new__(cls, sp.Symbol(str(head_name)), label_expr)

    @property
    def head_name(self) -> str:
        return str(self.args[0])

    @property
    def head(self) -> "IndexType":
        return _INDEX_TYPES[self.head_name]

    @property
    def label(self):
        label_expr = self.args[1]
        if label_expr.is_Integer:
            return int(label_expr)
        if isinstance(label_expr, sp.Wild):
            return label_expr
        return str(label_expr)

    def with_label(self, label) -> "Index":
        return Index(self.head_name, label)

    def dual(self) -> "Index":
        return self.head.dual(self.label)

    def _sympystr(self, printer):
        return f"{self.head_name}({self.label!r})"


def _index_label_expr(label):
    if isinstance(label, int):
        return sp.Integer(label)
    if isinstance(label, sp.Integer):
        return label
    if isinstance(label, sp.Wild):
        return label
    return sp.Symbol(str(label))


@dataclass(frozen=True)
class IndexType:
    name: str
    variance: str
    dim: Any = None
    index_set: tuple = LatinIdx
    color: Any = "Black"
    dual_name: str | None = None
    explicit: bool = False

    def __call__(self, label) -> Index:
        return Index(self.name, label)

    def dual(self, label) -> Index:
        if self.dual_name is None:
            raise ValueError(f"Index type {self.name} has no dual.")
        return _INDEX_TYPES[self.dual_name](label)

    def with_metadata(self, *, dim, index_set, color, dual_name) -> "IndexType":
        return IndexType(
            self.name,
            self.variance,
            dim=dim,
            index_set=tuple(index_set),
            color=color,
            dual_name=dual_name,
            explicit=self.explicit,
        )


@dataclass(frozen=True)
class TensorHead:
    name: str

    @property
    def symbol(self) -> sp.Symbol:
        return sp.Symbol(self.name)

    def __call__(self, *args):
        return _make_tensor_call(self.symbol, *args)

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TensorSymmetry:
    slots: tuple[int, ...]


@dataclass(frozen=True)
class Symmetric(TensorSymmetry):
    pass


@dataclass(frozen=True)
class Antisymmetric(TensorSymmetry):
    pass


@dataclass(frozen=True)
class PermutationSymmetry(TensorSymmetry):
    pass


@dataclass(frozen=True)
class Cycles:
    cycles: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        object.__setattr__(self, "cycles", tuple(tuple(cycle) for cycle in self.cycles))


def _register_index_type(index_type: IndexType) -> IndexType:
    _INDEX_TYPES[index_type.name] = index_type
    target_list = IdxUpList if index_type.variance == "up" else IdxDnList
    for collection in (IdxList, target_list):
        for pos, existing in enumerate(collection):
            if existing.name == index_type.name:
                collection[pos] = index_type
                break
        else:
            collection.append(index_type)
    return index_type


UP = _register_index_type(IndexType("UP", "up", dim=DefaultDim, dual_name="DN"))
DN = _register_index_type(IndexType("DN", "down", dim=DefaultDim, dual_name="UP"))
UE = _register_index_type(IndexType("UE", "up", color="Gray", dual_name="DE", explicit=True))
DE = _register_index_type(IndexType("DE", "down", color="Gray", dual_name="UE", explicit=True))


def declare_idx(up, down=None, *, dim, index_set=LatinIdx, color="Black"):
    if down is None:
        up, down = up
    up_type = IndexType(str(up), "up", dim=sp.sympify(dim), index_set=tuple(index_set), color=color, dual_name=str(down))
    down_type = IndexType(
        str(down), "down", dim=sp.sympify(dim), index_set=tuple(index_set), color=color, dual_name=str(up)
    )
    _register_index_type(up_type)
    _register_index_type(down_type)
    _CONSTANTS.add(sp.sympify(dim))
    return up_type, down_type


def DeclareIdx(ids, dim, index_set=LatinIdx, color="Black"):
    return declare_idx(ids, dim=dim, index_set=index_set, color=color)


def declare_explicit_idx(up, down=None, *, color="Gray"):
    if down is None:
        up, down = up
    up_type = IndexType(str(up), "up", color=color, dual_name=str(down), explicit=True)
    down_type = IndexType(str(down), "down", color=color, dual_name=str(up), explicit=True)
    _register_index_type(up_type)
    _register_index_type(down_type)
    return up_type, down_type


def DeclareExplicitIdx(ids, color="Gray"):
    return declare_explicit_idx(ids, color=color)


def IdxDual(index_type):
    if isinstance(index_type, Index):
        index_type = index_type.head
    return _INDEX_TYPES[index_type.dual_name]


def IdxSet(index_type):
    if isinstance(index_type, Index):
        index_type = index_type.head
    return tuple(index_type.index_set)


def IdxColor(index_type):
    if isinstance(index_type, Index):
        index_type = index_type.head
    return index_type.color


def Dim(index_type):
    if isinstance(index_type, Index):
        index_type = index_type.head
    return index_type.dim


def IdxHeadPtn(value) -> bool:
    return isinstance(value, IndexType) and _is_registered_implicit_index_type(value)


def IdxPtn(value) -> bool:
    return isinstance(value, Index) and _is_registered_implicit_index_type(value.head)


def IdxUpPtn(value) -> bool:
    index_type = _coerce_index_or_type(value)
    return index_type is not None and index_type.variance == "up" and _is_registered_implicit_index_type(index_type)


def IdxDnPtn(value) -> bool:
    index_type = _coerce_index_or_type(value)
    return index_type is not None and index_type.variance == "down" and _is_registered_implicit_index_type(index_type)


def _coerce_index_or_type(value):
    if isinstance(value, Index):
        return value.head
    if isinstance(value, IndexType):
        return value
    return None


def _is_registered_implicit_index_type(index_type: IndexType) -> bool:
    return not index_type.explicit and any(existing.name == index_type.name for existing in IdxList)


def Uniq(n: int):
    return [f"uq{next(_UNIQ_COUNTER)}" for _ in range(int(n))]


def Uq(n: int):
    return tuple(Uniq(n))


def UniqueIdx():
    return Uniq(50)


class _Dta(sp.Function):
    nargs = 2

    @classmethod
    def eval(cls, left, right):
        if isinstance(left, Index) and isinstance(right, Index):
            if left.head.explicit and right.head.explicit:
                return sp.KroneckerDelta(left.label, right.label)
            if _same_declared_pair(left, right) and left.label == right.label:
                return sp.sympify(left.head.dim)
        return None


class _PdVars(sp.Function):
    nargs = None

    @classmethod
    def eval(cls, *args):
        ordered = tuple(sorted(args, key=_pdvars_sort_key))
        if args != ordered:
            return cls(*ordered)
        return None


class _PdT(sp.Function):
    nargs = 2


class _Pdts(sp.Function):
    nargs = None

    def _sympystr(self, printer):
        order, head, *indices = self.args
        indices_text = ", ".join(printer.doprint(index) for index in indices)
        return f"Pdts({printer.doprint(order)}, {printer.doprint(head)})({indices_text})"


class _Pm2(sp.Function):
    nargs = 2


class _LeviCivita(sp.Function):
    nargs = None

    @classmethod
    def eval(cls, *args):
        indices = [arg for arg in args if isinstance(arg, Index)]
        if _levicivita_has_declared_dimension(indices, args) and len(set(indices)) != len(indices):
            return sp.Integer(0)
        return None


class _TensorCall(sp.Function):
    nargs = None

    @classmethod
    def eval(cls, head, *args):
        if len(args) == 2 and all(isinstance(arg, Index) for arg in args):
            left, right = args
            if _is_registered_metric_delta_pair(head, left, right):
                return Dta(left, right)
        return None

    @property
    def head_symbol(self):
        return self.args[0]

    @property
    def tensor_args(self):
        return self.args[1:]

    def _sympystr(self, printer):
        args = ", ".join(printer.doprint(arg) for arg in self.tensor_args)
        return f"{self.head_symbol}({args})"


def _make_tensor_call(head_symbol, *args):
    sympy_args = tuple(sp.sympify(arg) for arg in args)
    return _TensorCall(head_symbol, *_orderless_tensor_args(head_symbol, sympy_args))


def _same_declared_pair(left: Index, right: Index) -> bool:
    return left.head_name == right.head_name or left.head.dual_name == right.head_name


def _is_registered_metric_delta_pair(head, left: Index, right: Index) -> bool:
    for up, down in _METRIC_INDEX_PAIRS.get(head, ()):
        if left.head_name == up.name and right.head_name == down.name:
            return True
        if left.head_name == down.name and right.head_name == up.name:
            return True
        if _is_total_metric_pair(up, down) and left.head.dual_name == right.head_name:
            return True
        if _is_total_metric_pair(up, down) and right.head.dual_name == left.head_name:
            return True
    return False


def _is_total_metric_pair(up: IndexType, down: IndexType) -> bool:
    return up.name == "UTot" and down.name == "DTot"


def _index_sort_key(index: Index):
    return (str(index.label), index.head_name)


def _pdvars_sort_key(value):
    if isinstance(value, Index):
        return (0, _index_sort_key(value))
    return (1, sp.default_sort_key(value))


def Dta(*args):
    if len(args) != 2:
        return _inert_dta(*(sp.sympify(arg) for arg in args))
    left, right = args
    left = sp.sympify(left)
    right = sp.sympify(right)
    if isinstance(left, Index) and isinstance(right, Index) and _index_sort_key(right) < _index_sort_key(left):
        left, right = right, left
    return _Dta(left, right)


def _inert_dta(*args):
    return _make_tensor_call(sp.Symbol("Dta"), *args)


def DtaGen(*indices, dta=Dta, DtaGenDta=None):
    if DtaGenDta is not None:
        dta = DtaGenDta
    if not indices:
        return _make_tensor_call(sp.Symbol("DtaGen"))
    if len(indices) % 2 != 0:
        raise ValueError("DtaGen requires an equal number of upper and lower indices.")
    degree = len(indices) // 2
    left = tuple(sp.sympify(index) for index in indices[:degree])
    right = tuple(sp.sympify(index) for index in indices[degree:])
    terms = []
    for permuted in permutations(range(degree)):
        sign = _permutation_signature(tuple(range(degree)), permuted)
        terms.append(sign * sp.Mul(*(dta(left[pos], right[permuted[pos]]) for pos in range(degree))))
    return sp.Add(*terms)


def LeviCivita(*indices):
    indices = tuple(sp.sympify(index) for index in indices)
    if not _levicivita_has_declared_dimension([index for index in indices if isinstance(index, Index)], indices):
        return _make_tensor_call(sp.Symbol("LeviCivita"), *indices)
    return _LeviCivita(*indices)


def _levicivita_has_declared_dimension(indices, args):
    if len(indices) != len(args) or not indices:
        return False
    first_head = indices[0].head_name
    if any(index.head_name != first_head for index in indices):
        return False
    dimension = sp.sympify(indices[0].head.dim)
    return bool(dimension.is_Integer and int(dimension) == len(indices))


def register_metric(metric, indices=(UP, DN)):
    up, down = indices
    head_symbol = _metric_head_symbol(metric)
    _METRIC_HEADS.add(head_symbol)
    _METRICS[up.name] = metric
    _METRICS[down.name] = metric
    pairs = _METRIC_INDEX_PAIRS.setdefault(head_symbol, [])
    if (up, down) not in pairs:
        pairs.append((up, down))


def _metric_head_symbol(metric) -> sp.Symbol:
    if isinstance(metric, TensorHead):
        return metric.symbol
    if isinstance(metric, sp.FunctionClass):
        return sp.Symbol(metric.__name__)
    return sp.Symbol(str(metric))


def PdVars(*indices):
    return _PdVars(*(sp.sympify(index) for index in indices))


def PdT(*args):
    if len(args) != 2:
        return _inert_pdt(*(sp.sympify(arg) for arg in args))
    expr, variables = args
    if not isinstance(variables, _PdVars):
        try:
            variables = PdVars(*variables)
        except TypeError:
            return _inert_pdt(sp.sympify(expr), sp.sympify(variables))
    expr = sp.sympify(expr)
    if len(variables.args) == 0:
        return expr
    if expr.is_number or _is_constant_scalar(expr) or isinstance(expr, _Dta):
        return sp.Integer(0)
    if isinstance(expr, _Pm2):
        reduced = _reduce_pdt_pm2_derivative(expr, variables)
        if reduced is not None:
            return reduced
    if isinstance(expr, (sp.Add, sp.Mul, sp.Pow, _PdT)):
        result = expr
        for index in variables.args:
            result = Pd(result, index)
        return result
    return _PdT(expr, variables)


def _inert_pdt(*args):
    return _make_tensor_call(sp.Symbol("PdT"), *args)


def P(*indices):
    variables = PdVars(*indices)

    def _partial(expr):
        return PdT(expr, variables)

    return _partial


def TensorReplace(expr_or_rule, rule=None):
    from .util import TReplace

    return TReplace(expr_or_rule, rule)


def Pdts(order, head, *indices):
    head_symbol = _tensor_head_symbol(head) if isinstance(head, (TensorHead, _TensorCall)) else sp.sympify(head)
    return _Pdts(sp.Integer(order), head_symbol, *(sp.sympify(index) for index in indices))


def Pm2(*args, index_type=None):
    if index_type is not None:
        if len(args) != 1:
            return _inert_pm2(*(sp.sympify(arg) for arg in args), sp.sympify(index_type))
        expr = args[0]
    elif len(args) == 2:
        expr, index_type = args
    else:
        return _inert_pm2(*(sp.sympify(arg) for arg in args))
    expr = sp.sympify(expr)
    try:
        index_type = _coerce_index_type(index_type)
    except (KeyError, TypeError):
        return _inert_pm2(expr, sp.sympify(index_type))
    reduced = _reduce_pm2_derivative(expr, index_type)
    if reduced is not None:
        return reduced
    if isinstance(expr, _PdT):
        base, variables = expr.args
        return PdT(Pm2(base, index_type), variables)
    expanded = _expand_pm2_low_degree_plus_power(expr)
    if isinstance(expanded, sp.Add):
        return sp.Add(*(Pm2(term, index_type) for term in expanded.args))
    coeff, rest = expanded.as_coeff_Mul()
    if coeff != 1:
        return coeff * Pm2(rest, index_type)
    constant_factor, dynamic_part = _pm2_constant_factor(expanded, index_type)
    if constant_factor != 1:
        return constant_factor * Pm2(dynamic_part, index_type)
    return _Pm2(expanded, sp.Symbol(index_type.name))


def _expand_pm2_low_degree_plus_power(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Pow) and _is_pm2_expandable_plus_power(expr):
        return sp.expand(expr)
    if isinstance(expr, sp.Mul):
        factors = list(expr.args)
        for pos, factor in enumerate(factors):
            if not _is_pm2_expandable_plus_power(factor):
                continue
            expanded_factor = sp.expand(factor)
            rest = sp.Mul(*(factors[:pos] + factors[pos + 1 :]))
            return sp.Add(*(term * rest for term in sp.Add.make_args(expanded_factor)))
    return expr


def _is_pm2_expandable_plus_power(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return True
    return (
        isinstance(expr, sp.Pow)
        and isinstance(expr.base, sp.Add)
        and expr.exp.is_Integer
        and 0 <= int(expr.exp) < 5
    )


def _inert_pm2(*args):
    return _make_tensor_call(sp.Symbol("Pm2"), *args)


def _coerce_index_type(index_type) -> IndexType:
    if isinstance(index_type, IndexType):
        return index_type
    if isinstance(index_type, sp.Symbol):
        return _INDEX_TYPES[str(index_type)]
    return _INDEX_TYPES[str(index_type)]


def _reduce_pm2_derivative(expr, index_type: IndexType):
    if not isinstance(expr, _PdT):
        return None
    base, variables = expr.args
    vars_list = list(variables.args)
    for first, left in enumerate(vars_list):
        if not _is_index_of_type(left, index_type):
            continue
        for second in range(first + 1, len(vars_list)):
            right = vars_list[second]
            if _is_index_of_type(right, index_type) and right.label == left.label:
                remaining = [var for pos, var in enumerate(vars_list) if pos not in {first, second}]
                return PdT(base, PdVars(*remaining))
    return None


def _reduce_pdt_pm2_derivative(expr, variables):
    inner, index_type = pm2_parts(expr)
    vars_list = list(variables.args)
    for first, left in enumerate(vars_list):
        if not _is_index_of_type(left, index_type):
            continue
        for second in range(first + 1, len(vars_list)):
            right = vars_list[second]
            if _is_index_of_type(right, index_type) and right.label == left.label:
                remaining = [var for pos, var in enumerate(vars_list) if pos not in {first, second}]
                return PdT(inner, PdVars(*remaining))
    return None


def _pm2_constant_factor(expr, index_type: IndexType):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    if len(factors) < 2:
        return sp.Integer(1), expr
    test_index = index_type("__mathgr_pm2_test__")
    constant_factors = []
    dynamic_factors = []
    for factor in factors:
        if _pd_is_zero_for_constant_factor(factor, test_index):
            constant_factors.append(factor)
        else:
            dynamic_factors.append(factor)
    if not constant_factors or not dynamic_factors:
        return sp.Integer(1), expr
    return sp.Mul(*constant_factors), sp.Mul(*dynamic_factors)


def _pd_is_zero_for_constant_factor(expr, index):
    derivative = Pd(expr, index)
    if derivative == 0:
        return True
    if not SimpHook:
        return False
    return _apply_simp_hooks(derivative, tuple(SimpHook)) == 0


def _is_index_of_type(value, index_type: IndexType) -> bool:
    return isinstance(value, Index) and value.head_name == index_type.name


def tensor(name: str):
    return TensorHead(name)


def tensor_head_name(expr) -> str | None:
    expr = sp.sympify(expr)
    if isinstance(expr, _TensorCall):
        return str(expr.head_symbol)
    return None


def tensor_args(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, _TensorCall):
        return tuple(expr.tensor_args)
    return ()


def DeclareSym(head, index_signature, symmetry):
    key = _symmetry_key(head, index_signature)
    implicit_signature = _implicit_signature_items(index_signature)
    symmetries = tuple(_expand_all_symmetry(item, implicit_signature) for item in _coerce_symmetries(symmetry))
    if not all(_symmetry_valid_for_signature(index_signature, item) for item in symmetries):
        return None
    existing = list(_SYMMETRIES.get(key, []))
    for item in symmetries:
        if item not in existing:
            existing.append(item)
    _SYMMETRIES[key] = existing
    return existing


def _coerce_symmetries(symmetry):
    if isinstance(symmetry, (TensorSymmetry, Cycles)) or _is_permutation_spec(symmetry):
        return (_coerce_symmetry(symmetry),)
    if isinstance(symmetry, (list, tuple)):
        return tuple(_coerce_symmetry(item) for item in symmetry)
    return (_coerce_symmetry(symmetry),)


def _is_permutation_spec(symmetry) -> bool:
    return isinstance(symmetry, (list, tuple)) and all(isinstance(slot, int) for slot in symmetry)


def _coerce_symmetry(symmetry):
    if isinstance(symmetry, TensorSymmetry):
        return symmetry
    if _is_permutation_spec(symmetry):
        return PermutationSymmetry(tuple(symmetry))
    return symmetry


def _expand_all_symmetry(symmetry, implicit_signature):
    if isinstance(symmetry, (Symmetric, Antisymmetric)) and _slots_are_all(symmetry.slots):
        return type(symmetry)(tuple(range(1, len(implicit_signature) + 1)))
    return symmetry


def _slots_are_all(slots) -> bool:
    return slots == "All" or slots == ("All",) or slots == ["All"]


def DeleteSym(head, index_signature):
    _SYMMETRIES.pop(_symmetry_key(head, index_signature), None)
    return None


def ShowSym(head, index_signature):
    return list(_SYMMETRIES.get(_symmetry_key(head, index_signature), []))


def _symmetry_valid_for_signature(index_signature, symmetry) -> bool:
    implicit_signature = _implicit_signature_items(index_signature)
    if isinstance(symmetry, (Symmetric, Antisymmetric)):
        selected = _signature_items_for_declared_slots(implicit_signature, symmetry.slots)
        return bool(selected) and len({_signature_item(item) for item in selected}) == 1
    if isinstance(symmetry, PermutationSymmetry):
        return _permutation_preserves_signature(implicit_signature, symmetry.slots)
    if isinstance(symmetry, Cycles):
        permutation = _permutation_from_cycles(symmetry.cycles, len(implicit_signature))
        return _permutation_preserves_signature(implicit_signature, permutation)
    return True


def _implicit_signature_items(index_signature):
    return tuple(
        item
        for item in index_signature
        if not (isinstance(item, Index) and item.head.explicit)
    )


def _signature_items_for_declared_slots(implicit_signature, declared_slots):
    selected = []
    for slot in declared_slots:
        if slot < 1 or slot > len(implicit_signature):
            return ()
        selected.append(implicit_signature[slot - 1])
    return tuple(selected)


def _permutation_preserves_signature(implicit_signature, permutation) -> bool:
    permutation = tuple(permutation)
    if sorted(permutation) != list(range(1, len(implicit_signature) + 1)):
        return False
    original = tuple(_signature_item(item) for item in implicit_signature)
    permuted = tuple(_signature_item(implicit_signature[source_slot - 1]) for source_slot in permutation)
    return permuted == original


def _symmetry_key(head, index_signature):
    return (_tensor_head_symbol(head), tuple(_signature_item(item) for item in index_signature))


def _tensor_head_symbol(head) -> sp.Symbol:
    if isinstance(head, TensorHead):
        return head.symbol
    if isinstance(head, _TensorCall):
        return head.head_symbol
    if isinstance(head, sp.Symbol):
        return head
    return sp.Symbol(str(head))


def _signature_item(item):
    if isinstance(item, IndexType):
        return ("type", item.name)
    if isinstance(item, Index):
        return ("explicit", item.head_name, item.label)
    return ("literal", sp.sympify(item))


def is_pdt(expr) -> bool:
    return isinstance(sp.sympify(expr), _PdT)


def pdt_parts(expr):
    expr = sp.sympify(expr)
    if not isinstance(expr, _PdT):
        raise TypeError("Expected PdT expression.")
    base, variables = expr.args
    return base, tuple(variables.args)


def pd2pdts(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, _PdT):
        base, variables = expr.args
        if isinstance(base, _TensorCall):
            return Pdts(len(variables.args), base.head_symbol, *base.tensor_args, *variables.args)
        return Pdts(len(variables.args), base, *variables.args)
    if isinstance(expr, (Index, _Pdts)) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(pd2pdts(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def pdts2pd(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, _Pdts):
        order = int(expr.args[0])
        head = expr.args[1]
        indices = tuple(expr.args[2:])
        tensor_indices = indices[:-order] if order else indices
        derivative_indices = indices[-order:] if order else ()
        base = _make_tensor_call(head, *tensor_indices) if tensor_indices else head
        return PdT(base, PdVars(*derivative_indices))
    if isinstance(expr, (Index, _PdT)) or not getattr(expr, "args", ()):
        return expr
    rewritten_args = tuple(pdts2pd(arg) for arg in expr.args)
    if rewritten_args == expr.args:
        return expr
    return expr.func(*rewritten_args)


def is_pm2(expr) -> bool:
    return isinstance(sp.sympify(expr), _Pm2)


def pm2_parts(expr):
    expr = sp.sympify(expr)
    if not isinstance(expr, _Pm2):
        raise TypeError("Expected Pm2 expression.")
    inner, index_type = expr.args
    return inner, _coerce_index_type(index_type)


def Pd(*args, avoid=()):
    if len(args) != 2:
        return _inert_pd(*(sp.sympify(arg) for arg in args))
    expr, index = args
    return _pd(expr, index, tuple(avoid))


def _inert_pd(*args):
    return _make_tensor_call(sp.Symbol("Pd"), *args)


def _pd(expr, index, avoid_exprs=()):
    expr = sp.sympify(expr)
    index = sp.sympify(index)
    if expr.is_number or _is_constant_scalar(expr):
        return sp.Integer(0)
    if isinstance(expr, _Dta):
        return sp.Integer(0)
    metric_derivative = _inverse_metric_derivative(expr, index, *avoid_exprs)
    if metric_derivative is not None:
        return metric_derivative
    if isinstance(expr, sp.Add):
        return sp.Add(*(_pd(arg, index, avoid_exprs) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        terms = []
        args = expr.args
        for pos, arg in enumerate(args):
            rest = sp.Mul(*(args[:pos] + args[pos + 1 :]), evaluate=False)
            derivative = _pd(arg, index, (rest,))
            if derivative != 0:
                terms.append(sp.Mul(*(args[:pos] + (derivative,) + args[pos + 1 :])))
        return sp.Add(*terms) if terms else sp.Integer(0)
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        return (
            base ** (exponent - 1) * exponent * _pd(base, index, avoid_exprs)
            + base**exponent * sp.log(base) * _pd(exponent, index, avoid_exprs)
        )
    if isinstance(expr, _PdT):
        base, variables = expr.args
        return PdT(base, PdVars(index, *variables.args))
    if isinstance(expr, _Pm2):
        inner, type_symbol = expr.args
        return Pm2(Pd(inner, index), _coerce_index_type(type_symbol))
    return PdT(expr, PdVars(index))


def _inverse_metric_derivative(expr, index, *avoid_exprs):
    if not isinstance(expr, _TensorCall) or len(expr.tensor_args) != 2 or not isinstance(index, Index):
        return None
    left, right = expr.tensor_args
    if not isinstance(left, Index) or not isinstance(right, Index):
        return None
    for up, down in _METRIC_INDEX_PAIRS.get(expr.head_symbol, []):
        if left.head_name != up.name or right.head_name != up.name or index.head_name != down.name:
            continue
        first_label, second_label = _fresh_metric_labels(up, expr, index, *avoid_exprs, count=2)
        return (
            -_make_tensor_call(expr.head_symbol, up(first_label), right)
            * _make_tensor_call(expr.head_symbol, up(second_label), left)
            * Pd(_make_tensor_call(expr.head_symbol, down(first_label), down(second_label)), index)
        )
    return None


def _fresh_metric_labels(index_type, *exprs, count):
    used = {index.label for expr in exprs for index in _iter_indices(sp.sympify(expr))}
    labels = []
    for label in index_type.index_set:
        if label in used or label in labels:
            continue
        labels.append(label)
        if len(labels) == count:
            return tuple(labels)
    raise ValueError(f"Not enough fresh labels available for index type {index_type.name}.")


def _is_constant_scalar(expr) -> bool:
    if expr in _CONSTANTS:
        return True
    if isinstance(expr, Index):
        return False
    if expr.free_symbols:
        return False
    return not any(isinstance(node, (Index, sp.FunctionClass)) for node in sp.preorder_traversal(expr))


def idx(expr) -> list:
    return [index.label for index in _iter_indices(sp.sympify(expr), include_explicit=False)]


def free(expr) -> list:
    labels = idx(expr)
    counts = Counter(labels)
    return [label for label in labels if counts[label] == 1]


def dummy(expr) -> list:
    labels = idx(expr)
    counts = Counter(labels)
    seen = set()
    result = []
    for label in labels:
        if counts[label] == 2 and label not in seen:
            result.append(label)
            seen.add(label)
    return result


def rmE(indices):
    return [
        index
        for index in indices
        if not (isinstance(index, Index) and index.head.explicit)
    ]


def Sym(*args):
    if len(args) == 0:
        return _inert_sym()
    if len(args) > 2:
        return _inert_sym(*(sp.sympify(arg) for arg in args))
    expr = args[0]
    indices = args[1] if len(args) == 2 else None
    expr = sp.sympify(expr)
    indices = _symmetry_indices(expr, indices)
    return sp.Add(*(_replace_indices(expr, dict(zip(indices, perm, strict=True))) for perm in permutations(indices)))


def AntiSym(*args):
    if len(args) == 0:
        return _inert_antisym()
    if len(args) > 2:
        return _inert_antisym(*(sp.sympify(arg) for arg in args))
    expr = args[0]
    indices = args[1] if len(args) == 2 else None
    expr = sp.sympify(expr)
    indices = _symmetry_indices(expr, indices)
    terms = []
    for perm in permutations(indices):
        sign = _permutation_signature(indices, perm)
        terms.append(sign * _replace_indices(expr, dict(zip(indices, perm, strict=True))))
    return sp.Add(*terms)


def _inert_sym(*args):
    return _make_tensor_call(sp.Symbol("Sym"), *args)


def _inert_antisym(*args):
    return _make_tensor_call(sp.Symbol("AntiSym"), *args)


def _symmetry_indices(expr, indices):
    if indices is not None:
        if isinstance(indices, (list, tuple, set, sp.Tuple)):
            return tuple(sp.sympify(index) for index in indices)
        return (sp.sympify(indices),)
    all_indices = list(_iter_indices(expr, include_explicit=False))
    counts = Counter(index.label for index in all_indices)
    return tuple(index for index in all_indices if counts[index.label] == 1)


def _permutation_signature(original, permuted) -> int:
    positions = {value: pos for pos, value in enumerate(original)}
    inversions = 0
    for left in range(len(permuted)):
        for right in range(left + 1, len(permuted)):
            if positions[permuted[left]] > positions[permuted[right]]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _replace_indices(expr, replacements):
    return _normalize_deltas(expr.xreplace(replacements))


def _iter_indices(expr, *, include_explicit=True):
    if isinstance(expr, Index):
        if include_explicit or not expr.head.explicit:
            yield expr
        return
    if isinstance(expr, sp.Pow) and expr.exp.is_Integer and expr.exp > 0:
        for _ in range(int(expr.exp)):
            yield from _iter_indices(expr.base, include_explicit=include_explicit)
        return
    args = getattr(expr, "args", ())
    if isinstance(expr, sp.Mul):
        args = sorted(args, key=_idx_factor_order)
    for arg in args:
        yield from _iter_indices(arg, include_explicit=include_explicit)


def _idx_factor_order(expr):
    if isinstance(expr, _PdT):
        return (2, sp.default_sort_key(expr))
    if isinstance(expr, sp.Pow) and any(True for _ in _iter_indices(expr.base)):
        return (1, sp.default_sort_key(expr))
    return (0, sp.default_sort_key(expr))


def Simp(expr, **_options):
    hooks = tuple(SimpHook) + tuple(_options.get("hooks") or _options.get("Hooks") or ())
    dummy_pool = _coerce_dummy_pool_option(_options.get("Dummy", _options.get("dummy")))
    if _simp_method(_options) == "fast":
        return _simp_fast_with_hooks(expr, hooks, dummy_pool)
    return _simp_with_hooks(expr, hooks, dummy_pool)


def _simp_method(options):
    method = options.get("Method", options.get("method", "Hybrid"))
    return str(method).lower()


def _coerce_dummy_pool_option(dummy_pool):
    if dummy_pool is None or dummy_pool == "Automatic":
        return None
    if callable(dummy_pool):
        dummy_pool = dummy_pool()
    if isinstance(dummy_pool, (str, sp.Symbol)):
        return (str(dummy_pool),)
    return tuple(str(label) if isinstance(label, sp.Symbol) else label for label in dummy_pool)


def _simp_fast_with_hooks(expr, hooks, dummy_pool=None):
    expr = _apply_simp_hooks(sp.sympify(expr), hooks)
    if _should_simp_into(expr):
        return expr.func(_simp_fast_with_hooks(expr.args[0], hooks, dummy_pool))
    if _should_simp_into_hold(expr):
        return _simp_hold_expr(expr, hooks, dummy_pool, _simp_fast_with_hooks)
    expr = _split_indexed_powers(sp.expand(expr))
    if _should_simp_into_power(expr):
        base, exponent = expr.args
        return _simp_fast_with_hooks(base, hooks, dummy_pool) ** _simp_fast_with_hooks(exponent, hooks, dummy_pool)
    if isinstance(expr, sp.Add):
        return sp.Add(*(_simp_fast_with_hooks(arg, hooks, dummy_pool) for arg in _selected_simp_terms(expr)))
    expr = _contract_levicivita(expr)
    if isinstance(expr, sp.Add):
        return sp.Add(*(_simp_fast_with_hooks(arg, hooks, dummy_pool) for arg in _selected_simp_terms(expr)))
    expr = _contract_deltas(expr)
    expr = _canonicalize_dummy(expr, dummy_pool)
    return _apply_simp_hooks(expr, hooks)


def _simp_with_hooks(expr, hooks, dummy_pool=None):
    expr = _apply_simp_hooks(sp.sympify(expr), hooks)
    if _should_simp_into(expr):
        return expr.func(_simp_with_hooks(expr.args[0], hooks, dummy_pool))
    if _should_simp_into_hold(expr):
        return _simp_hold_expr(expr, hooks, dummy_pool, _simp_with_hooks)
    expr = _split_indexed_powers(sp.expand(expr))
    expr = pdts2pd(expr)
    if _should_simp_into_power(expr):
        base, exponent = expr.args
        return _simp_with_hooks(base, hooks, dummy_pool) ** _simp_with_hooks(exponent, hooks, dummy_pool)
    if isinstance(expr, sp.Add):
        simplified = sp.Add(*(_simp_with_hooks(arg, hooks, dummy_pool) for arg in _selected_simp_terms(expr)))
        if hooks and _has_fresh_hook_dummy_label(simplified):
            return _canonicalize_additive_tensor_terms(simplified, dummy_pool)
        return simplified
    expr = _contract_levicivita(expr)
    if isinstance(expr, sp.Add):
        terms = (_simp_with_hooks(arg, hooks, dummy_pool) for arg in _selected_simp_terms(expr))
        return _canonicalize_additive_tensor_terms(sp.Add(*terms), dummy_pool)
    if _is_zero_by_declared_symmetry(expr):
        return sp.Integer(0)
    expr = _contract_deltas(expr)
    expr = _contract_metric_products(expr)
    expr = _canonicalize_declared_symmetries(_contract_deltas(expr))
    use_tensor_reduce = bool(hooks)
    expr = _canonicalize_tensor_product(expr, dummy_pool, brute_force=use_tensor_reduce)
    expr = _canonicalize_product_symmetry_dummies(expr, dummy_pool)
    expr = _canonicalize_dummy(expr, dummy_pool)
    expr = _canonicalize_tensor_product(expr, dummy_pool, brute_force=use_tensor_reduce)
    expr = _canonicalize_declared_symmetries(expr)
    return _apply_simp_hooks(expr, hooks)


def _should_simp_into(expr):
    return expr.func in SimpInto1 and len(expr.args) == 1 and any(True for _ in _iter_indices(expr.args[0]))


def _should_simp_into_hold(expr):
    return expr.func.__name__ in {"_PdHold", "_IdHold"} and bool(expr.args)


def _simp_hold_expr(expr, hooks, dummy_pool, simp_fn):
    if expr.func.__name__ == "_PdHold" and len(expr.args) == 2:
        return expr.func(simp_fn(expr.args[0], hooks, dummy_pool), expr.args[1])
    if expr.func.__name__ == "_IdHold" and len(expr.args) == 1:
        return expr.func(simp_fn(expr.args[0], hooks, dummy_pool))
    return expr


def _should_simp_into_power(expr):
    if not isinstance(expr, sp.Pow):
        return False
    base, exponent = expr.args
    if exponent == 2:
        return False
    if _free_labels(base):
        return False
    return any(True for _ in _iter_indices(base)) or any(True for _ in _iter_indices(exponent))


def _selected_simp_terms(expr):
    terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]
    selected = SimpSelect(terms)
    if selected is None:
        return []
    if isinstance(selected, (list, tuple, set, sp.Tuple)):
        return list(selected)
    return [selected]


def SimpUq(expr, **options):
    options = dict(options)
    options.setdefault("Dummy", UniqueIdx)
    return Simp(expr, **options)


def _apply_simp_hooks(expr, hooks):
    current = sp.sympify(expr)
    for _ in range(_MAX_HOOK_ITERATIONS):
        previous = current
        for hook in hooks:
            current = _apply_single_simp_hook(current, hook)
        if current == previous:
            return current
    return current


def _apply_single_simp_hook(expr, hook):
    if callable(hook):
        return _apply_callable_simp_hook(expr, hook)
    if isinstance(hook, dict):
        return _apply_simp_hook_rules(expr, hook.items())
    if _is_simp_hook_rule_pair(hook):
        return _apply_simp_hook_rules(expr, (hook,))
    if isinstance(hook, (list, tuple)):
        return _apply_simp_hook_rules(expr, hook)
    raise TypeError("SimpHook entries must be callables, dicts, or sequences of (old, new) pairs.")


def _apply_callable_simp_hook(expr, hook):
    current = sp.sympify(expr)
    if current.args:
        rewritten_args = tuple(_apply_callable_simp_hook(arg, hook) for arg in current.args)
        if rewritten_args != current.args:
            current = current.func(*rewritten_args)
    hooked = sp.sympify(hook(current))
    return _freshen_hook_result_dummies(hooked, current)


def _freshen_hook_result_dummies(result, source):
    result = sp.sympify(result)
    source = sp.sympify(source)
    if result == source:
        return result
    source_labels = {}
    used_labels = {}
    for index in _iter_indices(source, include_explicit=False):
        family = _index_family_key(index)
        source_labels.setdefault(family, set()).add(index.label)
        used_labels.setdefault(family, set()).add(index.label)
    for index in _iter_indices(result, include_explicit=False):
        family = _index_family_key(index)
        used_labels.setdefault(family, set()).add(index.label)

    indices = list(_iter_indices(result, include_explicit=False))
    counts = Counter(_dummy_index_key(index) for index in indices)
    key_map = {}
    for index in indices:
        key = _dummy_index_key(index)
        family, label = key
        if counts[key] != 2 or label in source_labels.get(family, set()) or key in key_map:
            continue
        new_label = _fresh_hook_dummy_label(used_labels.setdefault(family, set()))
        used_labels[family].add(new_label)
        key_map[key] = new_label
    if not key_map:
        return result
    return _replace_index_keys(result, key_map)


def _fresh_hook_dummy_label(used):
    while True:
        candidate = f"uq{next(_UNIQ_COUNTER)}"
        if candidate not in used:
            return candidate


def _is_simp_hook_rule_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    left, right = value
    return not (_looks_like_simp_hook_rule_pair(left) and _looks_like_simp_hook_rule_pair(right))


def _looks_like_simp_hook_rule_pair(value):
    return isinstance(value, (list, tuple)) and len(value) == 2


def _apply_simp_hook_rules(expr, rules):
    replacements = tuple((sp.sympify(old), new if callable(new) else sp.sympify(new)) for old, new in rules)
    exact = {
        old: new
        for old, new in replacements
        if not callable(new) and not old.has(sp.Wild)
    }
    current = sp.sympify(expr).xreplace(exact)
    for old, new in replacements:
        if callable(new):
            if old.has(sp.Wild):
                current = current.replace(old, lambda **matches: sp.sympify(new(**matches)))
            else:
                current = current.replace(lambda node, old=old: node == old, lambda node: sp.sympify(new(node)))
            continue
        if old.has(sp.Wild):
            current = current.replace(old, new)
    return current


def _split_indexed_powers(expr):
    if isinstance(expr, sp.Add):
        return sp.Add(*(_split_indexed_powers(arg) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        scalar_contraction_power = _split_scalar_contraction_power_product(expr)
        if scalar_contraction_power is not None:
            return scalar_contraction_power
        factors = list(expr.args)
        new_factors = []
        changed = False
        for pos, factor in enumerate(factors):
            rest = sp.Mul(*(factors[:pos] + factors[pos + 1 :]), evaluate=False)
            replacement = _split_indexed_power_factor(factor, rest)
            if replacement is None:
                new_factors.append(factor)
            else:
                changed = True
                new_factors.append(replacement)
        return sp.Mul(*new_factors) if changed else expr
    return _split_indexed_power_factor(expr, sp.Integer(1)) or expr


def _split_scalar_contraction_power_product(expr):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    powered_positions = []
    for pos, factor in enumerate(factors):
        if not isinstance(factor, sp.Pow) or not factor.exp.is_Integer or factor.exp <= 1:
            continue
        if not any(True for _ in _iter_indices(factor.base, include_explicit=False)):
            continue
        powered_positions.append(pos)
    if len(powered_positions) < 2:
        return None

    exponents = {int(factors[pos].exp) for pos in powered_positions}
    if len(exponents) != 1:
        return None
    exponent = exponents.pop()

    base_product = sp.Mul(*(factors[pos].base for pos in powered_positions), evaluate=False)
    base_indices = list(_iter_indices(base_product, include_explicit=False))
    base_counts = Counter(_dummy_index_key(index) for index in base_indices)
    if not base_counts or any(count != 2 for count in base_counts.values()):
        return None

    rest_factors = [factor for pos, factor in enumerate(factors) if pos not in set(powered_positions)]
    rest_keys = {_dummy_index_key(index) for factor in rest_factors for index in _iter_indices(factor, include_explicit=False)}
    if rest_keys & set(base_counts):
        return None

    keys = _ordered_unique(_dummy_index_key(index) for index in base_indices)
    assignments = _dummy_key_copy_assignments(keys, exponent, rest_keys, dummy_pool=None)
    if assignments is None:
        return None

    split_factors = []
    for copy_pos in range(exponent):
        split_factors.append(_replace_index_keys(base_product, assignments[copy_pos]))
    return sp.Mul(*(rest_factors + split_factors))


def _split_indexed_power_factor(factor, rest):
    if not isinstance(factor, sp.Pow) or not factor.exp.is_Integer or factor.exp <= 1:
        return None
    base = factor.base
    exponent = int(factor.exp)
    indices = list(_iter_indices(base, include_explicit=False))
    if not indices:
        return None
    base_free = _free_index_keys(base)
    rest_free = _free_index_keys(rest)
    if exponent == 2 and base_free:
        return None
    if base_free and (base_free & rest_free or exponent % 2):
        return None

    power_per_copy = 2 if base_free else 1
    copies = exponent // power_per_copy
    keys = _ordered_unique(_dummy_index_key(index) for index in indices)
    assignments = _dummy_key_copy_assignments(keys, copies, rest_free, dummy_pool=None)
    if assignments is None:
        return None

    factors = []
    for copy_pos in range(copies):
        copied = _replace_index_keys(base, assignments[copy_pos])
        factors.append(copied**power_per_copy if power_per_copy != 1 else copied)
    return sp.Mul(*factors)


def _dummy_key_copy_assignments(keys, copies, protected_keys, dummy_pool=None):
    pools = {}
    protected_by_family = {}
    for family, label in protected_keys:
        protected_by_family.setdefault(family, set()).add(label)
    for key in keys:
        family, _label = key
        index_type = _INDEX_TYPES[family[0]]
        if index_type.dual_name not in family and len(family) == 2:
            index_type = _INDEX_TYPES[family[1]]
        pools.setdefault(family, tuple(LatinIdx if dummy_pool is None else dummy_pool) if dummy_pool else tuple(index_type.index_set))
    assignments = [dict() for _ in range(copies)]
    used_by_family = {family: set(labels) for family, labels in protected_by_family.items()}
    for key in keys:
        family, old_label = key
        pool = pools.get(family, LatinIdx)
        used = used_by_family.setdefault(family, set())
        available = [label for label in pool if label not in used]
        if len(available) < copies:
            return None
        for copy_pos in range(copies):
            new_label = available[copy_pos]
            used.add(new_label)
            if old_label != new_label:
                assignments[copy_pos][key] = new_label
    return assignments


def _free_labels(expr):
    labels = [index.label for index in _iter_indices(sp.sympify(expr), include_explicit=False)]
    counts = Counter(labels)
    return {label for label, count in counts.items() if count == 1}


def _free_index_keys(expr):
    keys = [_dummy_index_key(index) for index in _iter_indices(sp.sympify(expr), include_explicit=False)]
    counts = Counter(keys)
    return {key for key, count in counts.items() if count == 1}


def _ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dummy_label_pool(indices):
    for index in indices:
        return tuple(index.head.index_set)
    return LatinIdx


def _is_zero_by_declared_symmetry(expr):
    if isinstance(expr, _TensorCall):
        return _tensor_call_zero_by_symmetry(expr)
    if isinstance(expr, _PdT):
        base, _variables = expr.args
        return _is_zero_by_declared_symmetry(base)
    if isinstance(expr, sp.Mul):
        return any(_is_zero_by_declared_symmetry(arg) for arg in expr.args) or _product_zero_by_cross_symmetry(expr)
    return False


def _product_zero_by_cross_symmetry(expr):
    if _product_zero_by_metric_riemann_symmetry(expr):
        return True
    factors = list(_cross_symmetry_tensor_factors(expr.args))
    if len(factors) < 2:
        return False
    index_counts = Counter(
        _dual_family_index_key(index)
        for index in _iter_indices(expr, include_explicit=False)
    )
    symmetric_pairs = [
        pair
        for factor in factors
        for pair in _declared_index_pairs_for_factor(factor, Symmetric)
    ]
    antisymmetric_pairs = [
        pair
        for factor in factors
        for pair in _declared_index_pairs_for_factor(factor, Antisymmetric)
    ]
    for symmetric_pair in symmetric_pairs:
        if any(index_counts[index_key] != 2 for index_key in symmetric_pair):
            continue
        if any(symmetric_pair == antisymmetric_pair for antisymmetric_pair in antisymmetric_pairs):
            return True
    return False


def _product_zero_by_metric_riemann_symmetry(expr):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    metric_factors = [factor for factor in factors if _is_registered_same_variance_metric_factor(factor)]
    riemann_factors = [factor for factor in factors if factor.func.__name__ == "_LowerRiemann"]
    if not metric_factors or not riemann_factors:
        return False

    index_counts = Counter(
        _dual_family_index_key(index)
        for index in _iter_indices(expr, include_explicit=False)
    )
    for metric_factor in metric_factors:
        metric_pair = tuple(metric_factor.tensor_args)
        for riemann_factor in riemann_factors:
            riemann_indices = tuple(riemann_factor.args)
            for riemann_pair in (riemann_indices[:2], riemann_indices[2:]):
                if _metric_pair_contracts_riemann_pair(metric_pair, riemann_pair) and all(
                    index_counts[_dual_family_index_key(index)] == 2 for index in riemann_pair
                ):
                    return True
    return False


def _is_registered_same_variance_metric_factor(factor):
    if not isinstance(factor, _TensorCall) or factor.head_symbol not in _METRIC_HEADS or len(factor.tensor_args) != 2:
        return False
    left, right = factor.tensor_args
    return isinstance(left, Index) and isinstance(right, Index) and left.head_name == right.head_name


def _metric_pair_contracts_riemann_pair(metric_pair, riemann_pair):
    left_metric, right_metric = metric_pair
    left_riemann, right_riemann = riemann_pair
    return (
        _dual_indices_share_label(left_metric, left_riemann)
        and _dual_indices_share_label(right_metric, right_riemann)
    ) or (
        _dual_indices_share_label(left_metric, right_riemann)
        and _dual_indices_share_label(right_metric, left_riemann)
    )


def _dual_indices_share_label(left, right):
    return (
        isinstance(left, Index)
        and isinstance(right, Index)
        and left.label == right.label
        and left.head.dual_name == right.head_name
    )


def _dual_family_index_key(index):
    dual_name = index.head.dual_name or index.head_name
    return frozenset((index.head_name, dual_name)), index.label


def _cross_symmetry_tensor_factors(factors):
    for factor in factors:
        if isinstance(factor, _TensorCall):
            yield factor
            continue
        if isinstance(factor, _PdT) and isinstance(factor.args[0], _TensorCall):
            yield factor.args[0]


def _declared_index_pairs_for_factor(factor, symmetry_type):
    signature = tuple(_signature_for_arg(arg) for arg in factor.tensor_args)
    symmetries = _SYMMETRIES.get((factor.head_symbol, signature), [])
    pairs = []
    for symmetry in symmetries:
        if not isinstance(symmetry, symmetry_type):
            continue
        slots = _actual_slots_for_declared_slots(factor.tensor_args, symmetry.slots)
        for left_slot, right_slot in combinations(slots, 2):
            left = factor.tensor_args[left_slot - 1]
            right = factor.tensor_args[right_slot - 1]
            if not isinstance(left, Index) or not isinstance(right, Index):
                continue
            if left.head.explicit or right.head.explicit:
                continue
            pairs.append(frozenset((_dual_family_index_key(left), _dual_family_index_key(right))))
    return pairs


def _tensor_call_zero_by_symmetry(expr):
    signature = tuple(_signature_for_arg(arg) for arg in expr.tensor_args)
    symmetries = _SYMMETRIES.get((expr.head_symbol, signature), [])
    if not symmetries:
        return False
    for sym in symmetries:
        if isinstance(sym, Antisymmetric):
            slots = _actual_slots_for_declared_slots(expr.tensor_args, sym.slots)
            args = [expr.tensor_args[slot - 1] for slot in slots]
            if len(set(args)) != len(args):
                return True
    labels_by_slot = {slot: expr.tensor_args[slot - 1].label for slot in range(1, len(expr.tensor_args) + 1) if isinstance(expr.tensor_args[slot - 1], Index)}
    symmetric_pairs = {
        tuple(sorted((left, right)))
        for sym in symmetries
        if isinstance(sym, Symmetric)
        for slots in (_actual_slots_for_declared_slots(expr.tensor_args, sym.slots),)
        for left in slots
        for right in slots
        if left < right
    }
    antisymmetric_pairs = {
        tuple(sorted((left, right)))
        for sym in symmetries
        if isinstance(sym, Antisymmetric)
        for slots in (_actual_slots_for_declared_slots(expr.tensor_args, sym.slots),)
        for left in slots
        for right in slots
        if left < right
    }
    for left, right in symmetric_pairs & antisymmetric_pairs:
        if labels_by_slot.get(left) == labels_by_slot.get(right):
            return True
    repeated_pairs = [
        tuple(slots)
        for _label, slots in _slots_by_label(labels_by_slot).items()
        if len(slots) == 2
    ]
    for sym_pair in symmetric_pairs:
        for antisym_pair in antisymmetric_pairs:
            if all(_crosses_pair(pair, sym_pair, antisym_pair) for pair in repeated_pairs):
                return True
    return False


def _slots_by_label(labels_by_slot):
    grouped = {}
    for slot, label in labels_by_slot.items():
        grouped.setdefault(label, []).append(slot)
    return grouped


def _crosses_pair(repeated_pair, sym_pair, antisym_pair):
    return (
        len(repeated_pair) == 2
        and repeated_pair[0] in sym_pair
        and repeated_pair[1] in antisym_pair
    ) or (
        len(repeated_pair) == 2
        and repeated_pair[1] in sym_pair
        and repeated_pair[0] in antisym_pair
    )


def _signature_for_arg(arg):
    if isinstance(arg, Index):
        if arg.head.explicit:
            return ("explicit", arg.head_name, arg.label)
        return ("type", arg.head_name)
    return ("literal", arg)


def _orderless_tensor_args(head_symbol, args):
    symmetries = _SYMMETRIES.get((head_symbol, tuple(_signature_for_arg(arg) for arg in args)), ())
    non_explicit_slots = _non_explicit_slots(args)
    all_non_explicit_slots = tuple(range(1, len(non_explicit_slots) + 1))
    if not any(isinstance(symmetry, Symmetric) and symmetry.slots == all_non_explicit_slots for symmetry in symmetries):
        return args
    sorted_args = list(args)
    selected = sorted((args[slot - 1] for slot in non_explicit_slots), key=_canonical_slot_key)
    for slot, value in zip(non_explicit_slots, selected, strict=True):
        sorted_args[slot - 1] = value
    return tuple(sorted_args)


def _actual_slots_for_declared_slots(args, declared_slots):
    non_explicit_slots = _non_explicit_slots(args)
    return tuple(
        non_explicit_slots[slot - 1]
        for slot in declared_slots
        if 1 <= slot <= len(non_explicit_slots)
    )


def _non_explicit_slots(args):
    return tuple(
        slot
        for slot, arg in enumerate(args, start=1)
        if not (isinstance(arg, Index) and arg.head.explicit)
    )


def _canonicalize_declared_symmetries(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, _TensorCall):
        return _canonicalize_tensor_call_declared_symmetries(expr)
    if isinstance(expr, _PdT):
        base, variables = expr.args
        canonical_base = _canonicalize_declared_symmetries(base)
        if canonical_base == 0:
            return sp.Integer(0)
        coeff, rest = canonical_base.as_coeff_Mul()
        if coeff != 1:
            return coeff * PdT(rest, variables)
        return PdT(canonical_base, variables)
    if isinstance(expr, sp.Add):
        return sp.Add(*(_canonicalize_declared_symmetries(arg) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        return sp.Mul(*(_canonicalize_declared_symmetries(arg) for arg in expr.args))
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        return _canonicalize_declared_symmetries(base) ** _canonicalize_declared_symmetries(exponent)
    if not expr.args:
        return expr
    new_args = tuple(_canonicalize_declared_symmetries(arg) for arg in expr.args)
    if new_args == expr.args:
        return expr
    return expr.func(*new_args)


def _canonicalize_tensor_call_declared_symmetries(expr):
    signature = tuple(_signature_for_arg(arg) for arg in expr.tensor_args)
    symmetries = _SYMMETRIES.get((expr.head_symbol, signature), [])
    if not symmetries:
        return expr
    sign = 1
    args = tuple(expr.tensor_args)
    for sym in symmetries:
        canonical = _canonicalize_slots_for_symmetry(args, sym)
        if canonical is None:
            return sp.Integer(0)
        sym_sign, args = canonical
        sign *= sym_sign
    result = _make_tensor_call(expr.head_symbol, *args)
    return result if sign == 1 else -result


def _canonicalize_slots_for_symmetry(args, symmetry):
    if isinstance(symmetry, PermutationSymmetry):
        return _canonicalize_slots_for_permutation(args, symmetry.slots)
    if isinstance(symmetry, Cycles):
        return _canonicalize_slots_for_permutation(args, _permutation_from_cycles(symmetry.cycles, len(_non_explicit_slots(args))))
    slots = _actual_slots_for_declared_slots(args, symmetry.slots)
    if len(slots) < 2:
        return 1, args
    selected = tuple(args[slot - 1] for slot in slots)
    if isinstance(symmetry, Antisymmetric) and len(set(selected)) != len(selected):
        return None

    candidates = []
    for permuted_slots in permutations(slots):
        candidate_args = list(args)
        for target_slot, source_slot in zip(slots, permuted_slots, strict=True):
            candidate_args[target_slot - 1] = args[source_slot - 1]
        candidate_sign = _permutation_signature(slots, permuted_slots) if isinstance(symmetry, Antisymmetric) else 1
        candidates.append((tuple(_canonical_slot_key(arg) for arg in candidate_args), candidate_sign, tuple(candidate_args)))
    _key, candidate_sign, candidate_args = min(candidates, key=lambda item: (item[0], item[1]))
    return candidate_sign, candidate_args


def _canonical_slot_key(arg):
    if isinstance(arg, Index):
        return (0, _index_sort_key(arg))
    return (1, sp.default_sort_key(arg))


def _canonicalize_slots_for_permutation(args, permutation):
    permutation = tuple(permutation)
    actual_slots = _non_explicit_slots(args)
    if sorted(permutation) != list(range(1, len(actual_slots) + 1)):
        return 1, args
    candidates = []
    current = tuple(range(1, len(actual_slots) + 1))
    seen = set()
    while current not in seen:
        seen.add(current)
        candidate_args = list(args)
        for target_slot, source_logical_slot in zip(actual_slots, current, strict=True):
            source_actual_slot = actual_slots[source_logical_slot - 1]
            candidate_args[target_slot - 1] = args[source_actual_slot - 1]
        candidate_args = tuple(candidate_args)
        candidates.append((tuple(_canonical_slot_key(arg) for arg in candidate_args), candidate_args))
        current = tuple(current[source_slot - 1] for source_slot in permutation)
    _key, candidate_args = min(candidates, key=lambda item: item[0])
    return 1, candidate_args


def _canonicalize_product_symmetry_dummies(expr, dummy_pool=None):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return sp.Add(*(_canonicalize_product_symmetry_dummies(arg, dummy_pool) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        return _canonicalize_mul_symmetry_dummies(expr, dummy_pool)
    return expr


def _canonicalize_mul_symmetry_dummies(expr, dummy_pool=None):
    pdt_nodes = [arg for arg in expr.args if isinstance(arg, _PdT)]
    if len(pdt_nodes) != sum(1 for node in sp.preorder_traversal(expr) if isinstance(node, _PdT)):
        return expr
    variant_lists = [_factor_symmetry_variants(arg) for arg in expr.args]
    pdt_variant_factors = [
        factor
        for factor, variants in zip(expr.args, variant_lists, strict=True)
        if isinstance(factor, _PdT) and len(variants) > 1
    ]
    if len(pdt_variant_factors) > 1 and any(_pdt_base_is_registered_metric(factor) for factor in pdt_variant_factors):
        return expr
    if all(len(variants) == 1 for variants in variant_lists):
        return expr
    variant_count = 1
    for variants in variant_lists:
        variant_count *= len(variants)
        if variant_count > _MAX_PRODUCT_SYMMETRY_VARIANTS:
            return _canonicalize_large_mul_symmetry_dummies(expr, dummy_pool, variant_lists)
    candidates = []
    for variant_tuple in product(*variant_lists):
        sign = 1
        factors = []
        for factor_sign, factor in variant_tuple:
            sign *= factor_sign
            factors.append(factor)
        candidate = sp.Mul(sp.Integer(sign), *factors)
        candidates.append(_canonicalize_dummy(candidate, dummy_pool))
    if _contains_opposite_signed_candidates(candidates):
        return sp.Integer(0)
    return min(_unique_exprs(candidates), key=sp.default_sort_key)


def _canonicalize_mul_symmetry_dummies_for_add(expr, dummy_pool=None):
    pdt_nodes = [arg for arg in expr.args if isinstance(arg, _PdT)]
    if len(pdt_nodes) != sum(1 for node in sp.preorder_traversal(expr) if isinstance(node, _PdT)):
        candidate = _canonicalize_dummy_structural(expr, dummy_pool)
        return _canonicalize_dummy_by_renaming(candidate, dummy_pool)
    variant_lists = [_factor_symmetry_variants(arg) for arg in expr.args]
    pdt_variant_factors = [
        factor
        for factor, variants in zip(expr.args, variant_lists, strict=True)
        if isinstance(factor, _PdT) and len(variants) > 1
    ]
    if len(pdt_variant_factors) > 1 and any(_pdt_base_is_registered_metric(factor) for factor in pdt_variant_factors):
        candidate = _canonicalize_dummy_structural(expr, dummy_pool)
        return _canonicalize_dummy_by_renaming(candidate, dummy_pool)
    variant_count = 1
    for variants in variant_lists:
        variant_count *= len(variants)
        if variant_count > _MAX_PRODUCT_SYMMETRY_VARIANTS:
            candidate = _canonicalize_mul_symmetry_dummies(expr, dummy_pool)
            return _canonicalize_dummy_by_renaming(candidate, dummy_pool)
    candidates = []
    for variant_tuple in product(*variant_lists):
        sign = 1
        factors = []
        for factor_sign, factor in variant_tuple:
            sign *= factor_sign
            factors.append(factor)
        candidate = sp.Mul(sp.Integer(sign), *factors)
        candidate = _canonicalize_dummy_structural(candidate, dummy_pool)
        candidates.append(_canonicalize_dummy_by_renaming(candidate, dummy_pool))
    if _contains_opposite_signed_candidates(candidates):
        return sp.Integer(0)
    return min(_unique_exprs(candidates), key=sp.default_sort_key)


def _canonicalize_large_mul_symmetry_dummies(expr, dummy_pool, variant_lists):
    current = _canonicalize_dummy(expr, dummy_pool)
    if _large_mul_has_opposite_signed_factor_pair(current, dummy_pool):
        return sp.Integer(0)
    max_passes = 1 + sum(1 for variants in variant_lists if len(variants) > 1)
    for _ in range(max_passes):
        candidates = [current]
        factors = list(current.args) if isinstance(current, sp.Mul) else [current]
        for pos, factor in enumerate(factors):
            variants = _factor_symmetry_variants(factor)
            if len(variants) == 1:
                continue
            for sign, variant in variants:
                new_factors = list(factors)
                new_factors[pos] = variant
                candidate = sp.Mul(sp.Integer(sign), *new_factors)
                candidates.append(_canonicalize_dummy(candidate, dummy_pool))
        if _contains_opposite_signed_candidates(candidates):
            return sp.Integer(0)
        best = min(_unique_exprs(candidates), key=sp.default_sort_key)
        if best == current:
            return current
        current = best
    return current


def _large_mul_has_opposite_signed_factor_pair(expr, dummy_pool):
    factors = list(expr.args) if isinstance(expr, sp.Mul) else [expr]
    variant_factors = []
    for pos, factor in enumerate(factors):
        variants = _factor_symmetry_variants(factor)
        if len(variants) > 1:
            variant_factors.append((pos, variants))
    for left_index, (left_pos, left_variants) in enumerate(variant_factors):
        for right_pos, right_variants in variant_factors[left_index + 1 :]:
            candidates = []
            for left_sign, left_factor in left_variants:
                for right_sign, right_factor in right_variants:
                    new_factors = list(factors)
                    new_factors[left_pos] = left_factor
                    new_factors[right_pos] = right_factor
                    candidate = sp.Mul(sp.Integer(left_sign * right_sign), *new_factors)
                    candidates.append(_canonicalize_dummy(candidate, dummy_pool))
            if _contains_opposite_signed_candidates(candidates):
                return True
    return False


def _contains_opposite_signed_candidates(candidates):
    seen = set()
    for candidate in candidates:
        if -candidate in seen:
            return True
        seen.add(candidate)
    return False


def _pdt_base_is_registered_metric(expr):
    if not isinstance(expr, _PdT):
        return False
    base, _variables = expr.args
    return isinstance(base, _TensorCall) and base.head_symbol in _METRIC_HEADS


def _factor_symmetry_variants(factor):
    factor = sp.sympify(factor)
    if isinstance(factor, _TensorCall):
        return _tensor_call_symmetry_variants(factor)
    if isinstance(factor, _PdT):
        return _pdt_symmetry_variants(factor)
    return ((1, factor),)


def _pdt_symmetry_variants(expr):
    base, variables = expr.args
    if not isinstance(base, _TensorCall):
        return ((1, expr),)
    return tuple((sign, PdT(variant, variables)) for sign, variant in _tensor_call_symmetry_variants(base))


def _tensor_call_symmetry_variants(expr):
    signature = tuple(_signature_for_arg(arg) for arg in expr.tensor_args)
    symmetries = _SYMMETRIES.get((expr.head_symbol, signature), [])
    if not symmetries:
        return ((1, expr),)
    variants = {(1, tuple(expr.tensor_args))}
    for symmetry in symmetries:
        next_variants = set()
        for sign, args in variants:
            for symmetry_sign, candidate_args in _slot_symmetry_variants(args, symmetry):
                next_variants.add((sign * symmetry_sign, candidate_args))
        variants = next_variants
    return tuple(
        (sign, _TensorCall(expr.head_symbol, *args))
        for sign, args in sorted(variants, key=lambda item: (tuple(_canonical_slot_key(arg) for arg in item[1]), item[0]))
    )


def _slot_symmetry_variants(args, symmetry):
    if isinstance(symmetry, PermutationSymmetry):
        return _slot_permutation_symmetry_variants(args, symmetry.slots)
    if isinstance(symmetry, Cycles):
        return _slot_permutation_symmetry_variants(
            args,
            _permutation_from_cycles(symmetry.cycles, len(_non_explicit_slots(args))),
        )
    slots = _actual_slots_for_declared_slots(args, symmetry.slots)
    if len(slots) < 2:
        return ((1, args),)
    variants = []
    for permuted_slots in permutations(slots):
        candidate_args = list(args)
        for target_slot, source_slot in zip(slots, permuted_slots, strict=True):
            candidate_args[target_slot - 1] = args[source_slot - 1]
        sign = _permutation_signature(slots, permuted_slots) if isinstance(symmetry, Antisymmetric) else 1
        variants.append((sign, tuple(candidate_args)))
    return tuple(variants)


def _slot_permutation_symmetry_variants(args, permutation):
    permutation = tuple(permutation)
    actual_slots = _non_explicit_slots(args)
    if sorted(permutation) != list(range(1, len(actual_slots) + 1)):
        return ((1, args),)
    variants = []
    current = tuple(range(1, len(actual_slots) + 1))
    seen = set()
    while current not in seen:
        seen.add(current)
        candidate_args = list(args)
        for target_slot, source_logical_slot in zip(actual_slots, current, strict=True):
            source_actual_slot = actual_slots[source_logical_slot - 1]
            candidate_args[target_slot - 1] = args[source_actual_slot - 1]
        variants.append((1, tuple(candidate_args)))
        current = tuple(current[source_slot - 1] for source_slot in permutation)
    return tuple(variants)


def _canonicalize_additive_tensor_terms(expr, dummy_pool=None):
    expr = sp.sympify(expr)
    if not isinstance(expr, sp.Add):
        return _canonicalize_tensor_product(expr, dummy_pool)
    buckets = {}
    representatives = {}
    entry_counts = {}
    for raw_term in sp.Add.make_args(expr):
        representative = _canonicalize_tensor_product(raw_term, dummy_pool)
        term = _canonicalize_tensor_product(raw_term, dummy_pool, brute_force=True)
        coeff, rest = term.as_coeff_Mul()
        rest = _canonicalize_tensor_product(rest, dummy_pool, brute_force=True)
        buckets[rest] = buckets.get(rest, sp.Integer(0)) + coeff
        representatives.setdefault(rest, representative)
        entry_counts[rest] = entry_counts.get(rest, 0) + 1
    terms = []
    for rest, coeff in buckets.items():
        if coeff == 0:
            continue
        if entry_counts[rest] == 1 and _has_registered_metric_derivative_factor(representatives[rest]):
            terms.append(representatives[rest])
        else:
            terms.append(coeff * rest)
    return sp.Add(*terms)


def _canonicalize_tensor_product(term, dummy_pool=None, *, brute_force=False):
    term = sp.sympify(term)
    if isinstance(term, sp.Add):
        return _canonicalize_additive_tensor_terms(term, dummy_pool)
    if not _contains_implicit_index(term):
        return term
    if isinstance(term, sp.Mul):
        if brute_force:
            return _canonicalize_mul_symmetry_dummies_for_add(term, dummy_pool)
        if _has_registered_metric_derivative_factor(term):
            return term
        term = _canonicalize_mul_symmetry_dummies(term, dummy_pool)
    if brute_force:
        term = _canonicalize_dummy_structural(term, dummy_pool)
        term = _canonicalize_dummy_by_renaming(term, dummy_pool)
        return term
    term = _canonicalize_dummy(term, dummy_pool)
    return term


def _contains_implicit_index(expr):
    return any(True for _ in _iter_indices(sp.sympify(expr), include_explicit=False))


def _has_fresh_hook_dummy_label(expr):
    return any(
        isinstance(index.label, str) and index.label.startswith("uq")
        for index in _iter_indices(sp.sympify(expr), include_explicit=False)
    )


def _has_registered_metric_derivative_factor(expr):
    for node in sp.preorder_traversal(sp.sympify(expr)):
        if not isinstance(node, _PdT):
            continue
        base, _variables = node.args
        if isinstance(base, _TensorCall) and base.head_symbol in _METRIC_HEADS:
            return True
    return False


def _unique_exprs(exprs):
    unique = []
    seen = set()
    for expr in exprs:
        if expr in seen:
            continue
        seen.add(expr)
        unique.append(expr)
    return unique


def _permutation_from_cycles(cycles, degree):
    source_order = list(range(1, degree + 1))
    for cycle in cycles:
        cycle = tuple(cycle)
        if len(cycle) < 2:
            continue
        if any(slot < 1 or slot > degree for slot in cycle):
            continue
        for old_slot, new_slot in zip(cycle, cycle[1:] + cycle[:1], strict=True):
            source_order[new_slot - 1] = old_slot
    return tuple(source_order)


def _contract_levicivita(term):
    current = term
    changed = True
    while changed:
        changed = False
        factors = list(current.args) if isinstance(current, sp.Mul) else [current]
        for left_pos, left_factor in enumerate(factors):
            if not isinstance(left_factor, _LeviCivita):
                continue
            for right_pos in range(left_pos + 1, len(factors)):
                right_factor = factors[right_pos]
                if not isinstance(right_factor, _LeviCivita):
                    continue
                replacement = _levicivita_product(left_factor, right_factor)
                if replacement is None:
                    continue
                rest = [factor for pos, factor in enumerate(factors) if pos not in {left_pos, right_pos}]
                current = sp.expand(sp.Mul(*(rest + [replacement])))
                changed = True
                break
            if changed:
                break
    return current


def _levicivita_product(left, right):
    left_indices = tuple(left.args)
    right_indices = tuple(right.args)
    if len(left_indices) != len(right_indices) or not left_indices:
        return None
    if not all(isinstance(index, Index) for index in left_indices + right_indices):
        return None
    if not all(index.head_name == left_indices[0].head_name for index in left_indices):
        return None
    if not all(index.head_name == right_indices[0].head_name for index in right_indices):
        return None
    left_head = left_indices[0].head
    right_head = right_indices[0].head
    if left_head.dual_name == right_head.name:
        return DtaGen(*(left_indices + right_indices))
    if left_head.name == right_head.name and left_head.name in _METRICS:
        metric = _METRICS[left_head.name]
        return DtaGen(*(left_indices + right_indices), dta=metric)
    return None


def _contract_deltas(term):
    current = _expand_delta_power(term)
    changed = True
    while changed:
        changed = False
        factors = list(current.args) if isinstance(current, sp.Mul) else [current]
        for pos, factor in enumerate(factors):
            if not isinstance(factor, _Dta):
                continue
            left, right = factor.args
            if not isinstance(left, Index) or not isinstance(right, Index):
                continue
            rest_factors = factors[:pos] + factors[pos + 1 :]
            rest = sp.Mul(*rest_factors, evaluate=False) if rest_factors else sp.Integer(1)
            if _contains_index_key(rest, right):
                current = _replace_index_key_label(rest, right, left.label)
                changed = True
                break
            if _contains_index_key(rest, left):
                current = _replace_index_key_label(rest, left, right.label)
                changed = True
                break
    return current


def _contract_metric_products(term):
    current = term
    changed = True
    while changed:
        changed = False
        factors = list(current.args) if isinstance(current, sp.Mul) else [current]
        for left_pos, left_factor in enumerate(factors):
            if not _is_registered_two_index_metric(left_factor):
                continue
            for right_pos in range(left_pos + 1, len(factors)):
                right_factor = factors[right_pos]
                if not _is_registered_two_index_metric(right_factor):
                    continue
                replacement = _metric_product_delta(left_factor, right_factor)
                if replacement is None:
                    continue
                rest = [factor for pos, factor in enumerate(factors) if pos not in {left_pos, right_pos}]
                current = sp.Mul(*(rest + [replacement]))
                changed = True
                break
            if changed:
                break
    return current


def _is_registered_two_index_metric(factor) -> bool:
    return (
        isinstance(factor, _TensorCall)
        and factor.head_symbol in _METRIC_INDEX_PAIRS
        and len(factor.tensor_args) == 2
        and all(isinstance(arg, Index) for arg in factor.tensor_args)
    )


def _metric_product_delta(left, right):
    if left.head_symbol != right.head_symbol:
        return None
    for up, down in _METRIC_INDEX_PAIRS.get(left.head_symbol, []):
        left_delta = _metric_product_delta_for_pair(left, right, up, down)
        if left_delta is not None:
            return left_delta
        right_delta = _metric_product_delta_for_pair(right, left, up, down)
        if right_delta is not None:
            return right_delta
    return None


def _metric_product_delta_for_pair(inverse_metric, metric, up, down):
    inverse_args = tuple(inverse_metric.tensor_args)
    metric_args = tuple(metric.tensor_args)
    if not all(arg.head_name == up.name for arg in inverse_args):
        return None
    if not all(arg.head_name == down.name for arg in metric_args):
        return None
    for inverse_slot, inverse_index in enumerate(inverse_args):
        for metric_slot, metric_index in enumerate(metric_args):
            if inverse_index.label != metric_index.label:
                continue
            remaining_up = inverse_args[1 - inverse_slot]
            remaining_down = metric_args[1 - metric_slot]
            return Dta(remaining_up, remaining_down)
    return None


def _contains_label(expr, label) -> bool:
    return any(index.label == label for index in _iter_indices(expr))


def _replace_label(expr, old_label, new_label):
    return _replace_labels(expr, {old_label: new_label})


def _contains_index_key(expr, source_index) -> bool:
    source_key = _dummy_index_key(source_index)
    return any(_dummy_index_key(index) == source_key for index in _iter_indices(expr, include_explicit=False))


def _replace_index_key_label(expr, source_index, new_label):
    return _replace_index_keys(expr, {_dummy_index_key(source_index): new_label})


def _replace_labels(expr, label_map, *, include_explicit=True):
    replacements = {
        index: index.with_label(label_map[index.label])
        for index in set(_iter_indices(expr, include_explicit=include_explicit))
        if index.label in label_map
    }
    return _normalize_deltas(expr.xreplace(replacements))


def _expand_delta_power(expr):
    if isinstance(expr, sp.Pow) and isinstance(expr.base, _Dta) and expr.exp.is_Integer and expr.exp > 0:
        return sp.Mul(*[expr.base for _ in range(int(expr.exp))], evaluate=False)
    if isinstance(expr, sp.Mul):
        factors = list(expr.args)
        changed = False
        expanded_factors = []
        for pos, factor in enumerate(factors):
            if not (
                isinstance(factor, sp.Pow)
                and isinstance(factor.base, _Dta)
                and factor.exp.is_Integer
                and factor.exp > 0
            ):
                expanded_factors.append(factor)
                continue
            left, right = factor.base.args
            rest = sp.Mul(*(factors[:pos] + factors[pos + 1 :]), evaluate=False)
            if _contains_label(rest, left.label) or _contains_label(rest, right.label):
                expanded_factors.append(factor)
                continue
            expanded_factors.extend([factor.base] * int(factor.exp))
            changed = True
        if changed:
            return sp.Mul(*expanded_factors, evaluate=False)
    return expr


def _normalize_deltas(expr):
    if isinstance(expr, Index):
        return expr
    if isinstance(expr, _Dta):
        return Dta(*expr.args)
    if not getattr(expr, "args", ()):
        return expr
    new_args = tuple(_normalize_deltas(arg) for arg in expr.args)
    if new_args == expr.args:
        return expr
    return expr.func(*new_args)


def _canonicalize_dummy(term, dummy_pool=None):
    term = _split_overused_dummy_labels(term, dummy_pool)
    indices = list(_iter_indices(term, include_explicit=False))
    if not indices:
        return term
    keys = [_dummy_index_key(index) for index in indices]
    counts = Counter(keys)
    dummy_keys = {key for key, count in counts.items() if count == 2}
    if not dummy_keys:
        return term

    keys_by_family = {}
    protected_by_family = {}
    pools_by_family = {}
    for index in indices:
        family = _index_family_key(index)
        key = _dummy_index_key(index)
        pools_by_family.setdefault(family, tuple(index.head.index_set))
        if key in dummy_keys:
            family_keys = keys_by_family.setdefault(family, [])
            if key not in family_keys:
                family_keys.append(key)
        else:
            protected_by_family.setdefault(family, set()).add(index.label)

    configured_pool = tuple(LatinIdx if dummy_pool is None else dummy_pool)
    key_map = {}
    for family, ordered_keys in keys_by_family.items():
        pool = configured_pool if dummy_pool is not None else pools_by_family.get(family, LatinIdx)
        available = [label for label in pool if label not in protected_by_family.get(family, set())]
        for old_key, new_label in zip(ordered_keys, available, strict=False):
            if old_key[1] != new_label:
                key_map[old_key] = new_label
    if not key_map:
        return term
    return _replace_index_keys(term, key_map)


def _canonicalize_dummy_structural(term, dummy_pool=None):
    term = _split_overused_dummy_labels(term, dummy_pool)
    indices = list(_iter_indices(term, include_explicit=False))
    if not indices:
        return term
    keys = [_dummy_index_key(index) for index in indices]
    counts = Counter(keys)
    dummy_keys = {key for key, count in counts.items() if count == 2}
    if not dummy_keys:
        return term

    signatures = _dummy_key_signatures(term, dummy_keys)
    keys_by_family = {}
    protected_by_family = {}
    pools_by_family = {}
    for index in indices:
        family = _index_family_key(index)
        key = _dummy_index_key(index)
        pools_by_family.setdefault(family, tuple(index.head.index_set))
        if key in dummy_keys:
            keys_by_family.setdefault(family, set()).add(key)
        else:
            protected_by_family.setdefault(family, set()).add(index.label)

    configured_pool = tuple(LatinIdx if dummy_pool is None else dummy_pool)
    key_map = {}
    for family, family_keys in keys_by_family.items():
        ordered_keys = sorted(family_keys, key=lambda key: (signatures.get(key, ()), str(key[1])))
        pool = configured_pool if dummy_pool is not None else pools_by_family.get(family, LatinIdx)
        available = [label for label in pool if label not in protected_by_family.get(family, set())]
        for old_key, new_label in zip(ordered_keys, available, strict=False):
            if old_key[1] != new_label:
                key_map[old_key] = new_label
    if not key_map:
        return term
    return _replace_index_keys(term, key_map)


def _canonicalize_dummy_by_renaming(term, dummy_pool=None):
    indices = list(_iter_indices(term, include_explicit=False))
    if not indices:
        return term
    counts = Counter(_dummy_index_key(index) for index in indices)
    dummy_keys = {key for key, count in counts.items() if count == 2}
    if not dummy_keys:
        return term
    protected_by_family = {}
    keys_by_family = {}
    pools_by_family = {}
    for index in indices:
        family = _index_family_key(index)
        key = _dummy_index_key(index)
        pools_by_family.setdefault(family, tuple(index.head.index_set))
        if key in dummy_keys:
            keys_by_family.setdefault(family, set()).add(key)
        else:
            protected_by_family.setdefault(family, set()).add(index.label)

    family_specs = []
    variant_count = 1
    configured_pool = tuple(LatinIdx if dummy_pool is None else dummy_pool)
    for family, family_keys in sorted(keys_by_family.items(), key=lambda item: item[0]):
        ordered_keys = sorted(family_keys, key=lambda key: str(key[1]))
        pool = configured_pool if dummy_pool is not None else pools_by_family.get(family, LatinIdx)
        targets = [label for label in pool if label not in protected_by_family.get(family, set())][: len(ordered_keys)]
        if len(targets) < len(ordered_keys):
            return term
        if len(targets) > _MAX_FULL_DUMMY_RENAME_KEYS:
            return _canonicalize_dummy_by_tied_renaming(
                term,
                dummy_pool,
                dummy_keys,
                protected_by_family,
                keys_by_family,
                pools_by_family,
            )
        variant_count *= factorial(len(targets))
        if variant_count > _MAX_DUMMY_RENAME_VARIANTS:
            return _canonicalize_dummy_by_tied_renaming(
                term,
                dummy_pool,
                dummy_keys,
                protected_by_family,
                keys_by_family,
                pools_by_family,
            )
        family_specs.append((ordered_keys, targets))

    best = term
    family_permutations = (permutations(targets) for _keys, targets in family_specs)
    for choices in product(*family_permutations):
        key_map = {}
        for (ordered_keys, _targets), permuted_targets in zip(family_specs, choices, strict=True):
            for old_key, new_label in zip(ordered_keys, permuted_targets, strict=True):
                if old_key[1] != new_label:
                    key_map[old_key] = new_label
        candidate = _replace_index_keys(term, key_map) if key_map else term
        candidate = _canonicalize_declared_symmetries(candidate)
        if sp.default_sort_key(candidate) < sp.default_sort_key(best):
            best = candidate
    return best


def _canonicalize_dummy_by_tied_renaming(
    term,
    dummy_pool,
    dummy_keys,
    protected_by_family,
    keys_by_family,
    pools_by_family,
):
    signatures = _dummy_key_signatures(term, dummy_keys)
    family_variants = []
    variant_count = 1
    configured_pool = tuple(LatinIdx if dummy_pool is None else dummy_pool)
    for family, family_keys in sorted(keys_by_family.items(), key=lambda item: item[0]):
        ordered_keys = sorted(family_keys, key=lambda key: (signatures.get(key, ()), str(key[1])))
        pool = configured_pool if dummy_pool is not None else pools_by_family.get(family, LatinIdx)
        targets = [label for label in pool if label not in protected_by_family.get(family, set())][: len(ordered_keys)]
        if len(targets) < len(ordered_keys):
            return term
        grouped = []
        for key, target in zip(ordered_keys, targets, strict=True):
            if grouped and grouped[-1][0] == signatures.get(key, ()):
                grouped[-1][1].append(key)
                grouped[-1][2].append(target)
            else:
                grouped.append([signatures.get(key, ()), [key], [target]])
        group_options = [
            (keys, tuple(permutations(group_targets)))
            for _signature, keys, group_targets in grouped
        ]
        family_maps = []
        for choices in product(*(options for _keys, options in group_options)):
            key_map = {}
            for (keys, _options), group_targets in zip(group_options, choices, strict=True):
                for old_key, new_label in zip(keys, group_targets, strict=True):
                    if old_key[1] != new_label:
                        key_map[old_key] = new_label
            family_maps.append(key_map)
        variant_count *= len(family_maps)
        if variant_count > _MAX_DUMMY_RENAME_VARIANTS:
            return term
        family_variants.append(tuple(family_maps))

    best = term
    for choices in product(*family_variants):
        key_map = {}
        for family_map in choices:
            key_map.update(family_map)
        candidate = _replace_index_keys(term, key_map) if key_map else term
        candidate = _canonicalize_declared_symmetries(candidate)
        if sp.default_sort_key(candidate) < sp.default_sort_key(best):
            best = candidate
    return best


def _split_overused_dummy_labels(term, dummy_pool=None):
    records = _index_occurrence_records(term)
    if not records:
        return term
    has_fresh_hook_label = any(isinstance(record[1].label, str) and record[1].label.startswith("uq") for record in records)
    by_key = {}
    for record in records:
        _order, index, _context = record
        by_key.setdefault(_dummy_index_key(index), []).append(record)

    protected_by_family = {}
    pools_by_family = {}
    for key, key_records in by_key.items():
        family, label = key
        first_index = key_records[0][1]
        pools_by_family.setdefault(family, tuple(first_index.head.index_set))
        if len(key_records) <= 2:
            protected_by_family.setdefault(family, set()).add(label)

    occurrence_labels = {}
    used_by_family = {family: set(labels) for family, labels in protected_by_family.items()}
    configured_pool = tuple(LatinIdx if dummy_pool is None else dummy_pool)
    for key, key_records in by_key.items():
        if len(key_records) <= 2:
            continue
        family, label = key
        by_head = {}
        for record in key_records:
            by_head.setdefault(record[1].head_name, []).append(record)
        pool = configured_pool if dummy_pool is not None else pools_by_family.get(family, LatinIdx)
        protected = used_by_family.setdefault(family, set())
        if len(by_head) == 1:
            if not has_fresh_hook_label or len(key_records) % 2:
                continue
            available = [candidate for candidate in pool if candidate not in protected]
            pair_count = len(key_records) // 2
            if len(available) < pair_count:
                continue
            ordered = sorted(key_records, key=lambda record: (record[2], record[0]))
            for new_label, pos in zip(available, range(0, len(ordered), 2), strict=False):
                occurrence_labels[ordered[pos][0]] = new_label
                occurrence_labels[ordered[pos + 1][0]] = new_label
                protected.add(new_label)
            continue
        if len(by_head) != 2:
            continue
        left_head, right_head = sorted(by_head)
        left_records = by_head[left_head]
        right_records = by_head[right_head]
        if len(left_records) != len(right_records):
            continue
        left_index = left_records[0][1]
        right_index = right_records[0][1]
        if left_index.head.dual_name != right_head or right_index.head.dual_name != left_head:
            continue
        available = [candidate for candidate in pool if candidate not in protected]
        if len(available) < len(left_records):
            continue
        left_ordered = sorted(left_records, key=lambda record: (record[2], record[0]))
        right_ordered = sorted(right_records, key=lambda record: (record[2], record[0]))
        for new_label, left_record, right_record in zip(available, left_ordered, right_ordered, strict=False):
            occurrence_labels[left_record[0]] = new_label
            occurrence_labels[right_record[0]] = new_label
            protected.add(new_label)

    if not occurrence_labels:
        return term
    return _replace_index_occurrences(term, occurrence_labels)


def _index_occurrence_records(term):
    records = []
    order = 0
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for factor in factors:
        factor_shape = _expr_shape_key(factor)
        for index, context in _index_occurrence_records_for_expr(factor, factor_shape):
            records.append((order, index, context))
            order += 1
    return records


def _index_occurrence_records_for_expr(expr, factor_shape, path=()):
    expr = sp.sympify(expr)
    if isinstance(expr, Index):
        if not expr.head.explicit:
            yield expr, (factor_shape, path, _index_role_key("index"), expr.head_name)
        return
    if isinstance(expr, _TensorCall):
        for slot, arg in enumerate(expr.tensor_args, start=1):
            if isinstance(arg, Index):
                if not arg.head.explicit:
                    yield arg, (
                        factor_shape,
                        path,
                        _index_role_key("tensor-slot"),
                        str(expr.head_symbol),
                        slot,
                        arg.head_name,
                    )
            else:
                yield from _index_occurrence_records_for_expr(
                    arg,
                    factor_shape,
                    path + (("tensor-arg", str(expr.head_symbol), slot),),
                )
        return
    if isinstance(expr, _PdT):
        base, variables = expr.args
        if isinstance(base, _TensorCall):
            for slot, arg in enumerate(base.tensor_args, start=1):
                if isinstance(arg, Index):
                    if not arg.head.explicit:
                        yield arg, (
                            factor_shape,
                            path,
                            _index_role_key("pdt-base-slot"),
                            str(base.head_symbol),
                            slot,
                            arg.head_name,
                        )
                else:
                    yield from _index_occurrence_records_for_expr(
                        arg,
                        factor_shape,
                        path + (("pdt-base-arg", str(base.head_symbol), slot),),
                    )
        else:
            yield from _index_occurrence_records_for_expr(base, factor_shape, path + (("pdt-base",),))
        for arg in variables.args:
            if isinstance(arg, Index):
                if not arg.head.explicit:
                    yield arg, (factor_shape, path, _index_role_key("derivative-slot"), arg.head_name)
            else:
                yield from _index_occurrence_records_for_expr(arg, factor_shape, path + (("derivative-arg",),))
        return
    if isinstance(expr, _Dta):
        for slot, arg in enumerate(expr.args, start=1):
            if isinstance(arg, Index) and not arg.head.explicit:
                yield arg, (factor_shape, path, _index_role_key("delta-slot"), slot, arg.head_name)
        return
    for pos, arg in enumerate(getattr(expr, "args", ())):
        yield from _index_occurrence_records_for_expr(arg, factor_shape, path + ((expr.func.__name__, pos),))


def _replace_index_occurrences(expr, occurrence_labels):
    order = count()

    def rebuild(node):
        if isinstance(node, Index):
            if node.head.explicit:
                return node
            occurrence = next(order)
            if occurrence in occurrence_labels:
                return node.with_label(occurrence_labels[occurrence])
            return node
        if not getattr(node, "args", ()):
            return node
        new_args = tuple(rebuild(arg) for arg in node.args)
        if new_args == node.args:
            return node
        return node.func(*new_args)

    return _normalize_deltas(rebuild(sp.sympify(expr)))


def _dummy_index_key(index):
    return (_index_family_key(index), index.label)


def _index_family_key(index):
    dual_name = index.head.dual_name or index.head_name
    return tuple(sorted((index.head_name, dual_name)))


def _dummy_key_signatures(term, dummy_keys):
    signatures = {key: [] for key in dummy_keys}
    factors = list(term.args) if isinstance(term, sp.Mul) else [term]
    for factor in factors:
        factor_shape = _expr_shape_key(factor)
        for key, context in _index_contexts(factor, factor_shape):
            if key in signatures:
                signatures[key].append(context)
    return {key: tuple(sorted(contexts)) for key, contexts in signatures.items()}


def _index_contexts(expr, factor_shape, path=()):
    expr = sp.sympify(expr)
    if isinstance(expr, Index):
        yield _dummy_index_key(expr), (factor_shape, path, _index_role_key("index"), expr.head_name)
        return
    if isinstance(expr, _TensorCall):
        for slot, arg in enumerate(expr.tensor_args, start=1):
            if isinstance(arg, Index):
                yield _dummy_index_key(arg), (
                    factor_shape,
                    path,
                    _index_role_key("tensor-slot"),
                    str(expr.head_symbol),
                    slot,
                    arg.head_name,
                )
            else:
                yield from _index_contexts(arg, factor_shape, path + (("tensor-arg", str(expr.head_symbol), slot),))
        return
    if isinstance(expr, _PdT):
        base, variables = expr.args
        if isinstance(base, _TensorCall):
            for slot, arg in enumerate(base.tensor_args, start=1):
                if isinstance(arg, Index):
                    yield _dummy_index_key(arg), (
                        factor_shape,
                        path,
                        _index_role_key("pdt-base-slot"),
                        str(base.head_symbol),
                        slot,
                        arg.head_name,
                    )
                else:
                    yield from _index_contexts(arg, factor_shape, path + (("pdt-base-arg", str(base.head_symbol), slot),))
        else:
            yield from _index_contexts(base, factor_shape, path + (("pdt-base",),))
        for arg in variables.args:
            if isinstance(arg, Index):
                yield _dummy_index_key(arg), (factor_shape, path, _index_role_key("derivative-slot"), arg.head_name)
            else:
                yield from _index_contexts(arg, factor_shape, path + (("derivative-arg",),))
        return
    if isinstance(expr, _Dta):
        for slot, arg in enumerate(expr.args, start=1):
            if isinstance(arg, Index):
                yield _dummy_index_key(arg), (factor_shape, path, _index_role_key("delta-slot"), slot, arg.head_name)
        return
    for pos, arg in enumerate(getattr(expr, "args", ())):
        yield from _index_contexts(arg, factor_shape, path + ((expr.func.__name__, pos),))


def _index_role_key(role: str):
    ranks = {
        "tensor-slot": 0,
        "pdt-base-slot": 0,
        "delta-slot": 1,
        "index": 2,
        "derivative-slot": 3,
    }
    return (ranks.get(role, 9), role)


def _expr_shape_key(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, Index):
        return ("Index", expr.head_name)
    if isinstance(expr, _TensorCall):
        return ("Tensor", str(expr.head_symbol), tuple(_expr_shape_key(arg) for arg in expr.tensor_args))
    if isinstance(expr, _PdT):
        base, variables = expr.args
        return ("PdT", _expr_shape_key(base), tuple(sorted((_expr_shape_key(arg) for arg in variables.args))))
    if isinstance(expr, _Dta):
        return ("Dta", tuple(_expr_shape_key(arg) for arg in expr.args))
    if isinstance(expr, sp.Mul):
        return ("Mul", tuple(sorted((_expr_shape_key(arg) for arg in expr.args))))
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        return ("Pow", _expr_shape_key(base), sp.default_sort_key(exponent))
    if not getattr(expr, "args", ()):
        return ("Atom", sp.default_sort_key(expr))
    return (expr.func.__name__, tuple(_expr_shape_key(arg) for arg in expr.args))


def _replace_index_keys(expr, key_map):
    replacements = {
        index: index.with_label(key_map[_dummy_index_key(index)])
        for index in set(_iter_indices(expr, include_explicit=False))
        if _dummy_index_key(index) in key_map
    }
    return _normalize_deltas(expr.xreplace(replacements))
