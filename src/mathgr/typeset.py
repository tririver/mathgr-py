from __future__ import annotations

from datetime import datetime
import re

import sympy as sp

from .tensor import DE, Index, is_pdt, is_pm2, pdt_parts, pm2_parts, tensor_args, tensor_head_name


ToTeXHook = []
ToTeXTemplate = True
_MAX_HOOK_ITERATIONS = 10


def ToTeXString(expr):
    expr = _apply_tex_hooks(sp.sympify(expr))
    return DecorateTeXString(_tex(expr))


def ToTeX(expr):
    print(ToTeXString(expr))
    return None


def DecorateTeXString(text):
    text = _remove_waste(str(text))
    text = _replace_time_dot(text)
    text = _remove_text_and_curly(text)
    text = _break_line(text)
    text = _add_header_tail(text)
    return _final_cleanup(text)


def _apply_tex_hooks(expr):
    current = sp.sympify(expr)
    for _ in range(_MAX_HOOK_ITERATIONS):
        previous = current
        for hook in ToTeXHook:
            current = _apply_single_hook(current, hook)
        if current == previous:
            return current
    return current


def _apply_single_hook(expr, hook):
    if callable(hook):
        return _apply_callable_tex_hook(expr, hook)
    if isinstance(hook, dict):
        return _apply_hook_rules(expr, hook.items())
    if _is_hook_rule_pair(hook):
        return _apply_hook_rules(expr, (hook,))
    if isinstance(hook, (list, tuple)):
        return _apply_hook_rules(expr, hook)
    raise TypeError("ToTeXHook entries must be callables, dicts, or sequences of (old, new) pairs.")


def _apply_callable_tex_hook(expr, hook):
    current = sp.sympify(expr)
    if current.args:
        rewritten_args = tuple(_apply_callable_tex_hook(arg, hook) for arg in current.args)
        if rewritten_args != current.args:
            current = current.func(*rewritten_args)
    return sp.sympify(hook(current))


def _is_hook_rule_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    left, right = value
    return not (_looks_like_rule_pair(left) and _looks_like_rule_pair(right))


def _looks_like_rule_pair(value):
    return isinstance(value, (list, tuple)) and len(value) == 2


def _apply_hook_rules(expr, rules):
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


def _tex(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, Index):
        return _tex_label(expr.label)
    head_name = tensor_head_name(expr)
    if head_name is not None:
        return _tex_tensor_call(expr)
    if is_pdt(expr):
        return _tex_pdt(expr)
    if is_pm2(expr):
        inner, _index_type = pm2_parts(expr)
        return rf"\partial^{{-2}}\left({_tex(inner)}\right)"
    if expr.func.__name__ == "_LeviCivita":
        return "LeviCivita" + _index_suffix(expr.args)
    if expr.func.__name__ == "_Dta":
        return _tex_delta(expr)
    if expr.func.__name__ == "_LowerRiemann":
        return _tex_lower_riemann(expr.args)
    if expr.func.__name__ == "_CovDLowerRiemannAtom":
        return _tex_covd_lower_riemann(expr.args)
    if isinstance(expr, sp.Add):
        return _tex_add(expr)
    if isinstance(expr, sp.Mul):
        return " ".join(_tex_mul_factor(arg) for arg in expr.args)
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        return f"{_tex_group(base)}^{{{_tex(exponent)}}}"
    return sp.latex(expr)


def _tex_tensor_call(expr):
    head_name = tensor_head_name(expr)
    args = tensor_args(expr)
    if head_name is None:
        return sp.latex(expr)
    upper = "".join(_tex_label(arg.label) for arg in args if isinstance(arg, Index) and arg.head.variance == "up")
    lower = "".join(_tex_label(arg.label) for arg in args if isinstance(arg, Index) and arg.head.variance == "down")
    other_args = [arg for arg in args if not isinstance(arg, Index)]
    base = str(head_name)
    if other_args:
        base += r"\left(" + ", ".join(_tex(arg) for arg in other_args) + r"\right)"
    if upper:
        base += f"^{{{upper}}}"
    if lower:
        base += f"_{{{lower}}}"
    return base


def _tex_pdt(expr):
    base, derivative_indices = pdt_parts(expr)
    derivative_indices = tuple(derivative_indices)
    time_count = sum(1 for index in derivative_indices if index == DE(0))
    non_time = tuple(index for index in derivative_indices if index != DE(0))
    base_tex = _tex(base)
    if time_count and is_pm2(base):
        base_tex = "".join(r"\partial_{0} " for _ in range(time_count)) + base_tex
    elif time_count:
        base_tex = _tex_time_derivative(base_tex, time_count)
    if non_time:
        prefix = "".join(rf"\partial_{{{_tex(index)}}} " for index in non_time)
        return prefix + base_tex
    return base_tex


def _tex_time_derivative(base_tex, count):
    if count == 1:
        command = "dot"
    elif count == 2:
        command = "ddot"
    elif count == 3:
        command = "dddot"
    else:
        command = "partial_" + "0" * count
    if command.startswith("partial_"):
        return "".join(r"\partial_{0} " for _ in range(count)) + base_tex
    return rf"\{command}{{{base_tex}}}"


def _tex_delta(expr):
    ordered = sorted(expr.args, key=_delta_tex_sort_key)
    return r"\delta" + _ordered_index_suffix(ordered)


def _delta_tex_sort_key(index):
    if not isinstance(index, Index):
        return (2, _tex(index))
    variance_order = 0 if index.head.variance == "down" else 1
    return (variance_order, _tex_label(index.label), index.head_name)


def _tex_lower_riemann(indices):
    return "R" + _index_suffix(indices)


def _tex_covd_lower_riemann(indices):
    riemann_indices = tuple(indices[:4])
    derivative = indices[4]
    return rf"\nabla_{{{_tex(derivative)}}} " + _tex_lower_riemann(riemann_indices)


def _tex_add(expr):
    parts = []
    for pos, arg in enumerate(_tex_add_args(expr)):
        arg = sp.sympify(arg)
        if arg.could_extract_minus_sign():
            body = _tex(-arg)
            parts.append("-" + body)
        else:
            body = _tex(arg)
            parts.append(body if pos == 0 else "+" + body)
    return "".join(parts)


def _tex_add_args(expr):
    return sorted(expr.args, key=lambda arg: (sp.sympify(arg).is_number, sp.default_sort_key(arg)))


def _tex_mul_factor(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, sp.Add):
        return r"\left(" + _tex(expr) + r"\right)"
    return _tex(expr)


def _tex_group(expr):
    expr = sp.sympify(expr)
    if isinstance(expr, (sp.Symbol, sp.Integer, sp.Rational)) or isinstance(expr, Index) or tensor_head_name(expr):
        return _tex(expr)
    return r"\left(" + _tex(expr) + r"\right)"


def _index_suffix(indices):
    upper = "".join(_tex_label(index.label) for index in indices if isinstance(index, Index) and index.head.variance == "up")
    lower = "".join(_tex_label(index.label) for index in indices if isinstance(index, Index) and index.head.variance == "down")
    suffix = ""
    if upper:
        suffix += f"^{{{upper}}}"
    if lower:
        suffix += f"_{{{lower}}}"
    return suffix


def _ordered_index_suffix(indices):
    parts = []
    for index in indices:
        if not isinstance(index, Index):
            continue
        marker = "^" if index.head.variance == "up" else "_"
        parts.append(f"{marker}{{{_tex_label(index.label)}}}")
    return "".join(parts)


def _tex_label(label):
    if isinstance(label, str):
        return label
    return sp.latex(label)


def _remove_waste(text):
    return text.replace(r"\text{}", "{}").replace("){}^", ")^").replace(r"\partial {}_", r"\partial_")


def _remove_text_and_curly(text):
    current = str(text).replace("$", "")
    return re.sub(r"\\text\{([^{}]*)\}", r"\1", current)


def _replace_time_dot(text):
    if r"\partial^{-2}" in text or r"\partial ^{-2}" in text:
        return text
    return (
        text.replace(r"\partial_0\partial_0\partial_0", r"\dddot")
        .replace(r"\partial_0\partial_0", r"\ddot")
        .replace(r"\partial_0", r"\dot")
    )


def _break_line(text):
    text = text.replace("+", "\n + ")
    chars = []
    for pos, char in enumerate(text):
        if char == "-" and pos > 0 and text[pos - 1] not in {"\n", "{"}:
            chars.append("\n - ")
        else:
            chars.append(char)
    return "".join(chars).replace("=", "\n= ").replace(r"\to", "\n \\to ")


def _add_header_tail(text):
    if not ToTeXTemplate:
        return text
    return (
        "%Generated by MathGR/typeset.m, "
        + datetime.now().strftime("%a %d %b %Y %H:%M:%S")
        + ".\n\\documentclass{revtex4}\n\\usepackage{breqn}\n\\begin{document}\n\\begin{dmath}\n"
        + text
        + "\n\\end{dmath}\n\\end{document}\n"
    )


def _final_cleanup(text):
    current = str(text)
    while "\n\n" in current:
        current = current.replace("\n\n", "\n")
    return current
