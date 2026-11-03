"""
formulation_text.py
===================
The formula text of an objective or a constraint ("Mass()", "MaterialVolumeFraction(2, tau=0.5)")
read into a quantity's name and options: parse_expression, and to_params_form for what
to_params holds.
"""
import re


def parse_expression(text):
    """'Name(a, b=1)' -> ('Name', [a], {'b': 1}). Only a single quantity is accepted."""
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\(([^()]*)\)\s*", text or "")
    if not m:
        raise ValueError(f"expression {text!r}: write one quantity, e.g. 'Mass()'; "
                         "sums and products of quantities are not supported yet")
    name, inside = m.group(1), m.group(2).strip()
    args, kwargs = [], {}
    for part in [p.strip() for p in inside.split(",")] if inside else []:
        key, eq, value = part.partition("=")
        try:
            v = float(value if eq else key)
        except ValueError:
            raise ValueError(f"expression {text!r}: argument {part!r} is not a number") from None
        if eq:
            kwargs[key.strip()] = v
        else:
            args.append(v)
    return name, args, kwargs


def to_params_form(text, role, bound=None):
    """What to_params holds for an expression: (name, options) for the objective, (name, options,
    bound) for a constraint; name is the quantity's, options None for a quantity that takes no
    arguments, else {argument: value} in the order the quantity declares them."""
    from lsr_mmto.formulation.quantities import QUANTITIES
    name, args, kwargs = parse_expression(text)
    by_name = {q.name: q for q in QUANTITIES}
    if name not in by_name:
        raise ValueError(f"expression {text!r}: unknown quantity {name!r}; "
                         f"known: {', '.join(by_name)}")
    q = by_name[name]
    if not (q.objective if role == "objective" else q.constraint):
        raise ValueError(f"{name} cannot be the {role}")
    if not q.arguments:
        if args or kwargs:
            raise ValueError(f"expression {text!r}: {name} takes no arguments")
        opt = None
    else:
        required = [a for a in q.arguments if a[2] is None]
        optional = [a for a in q.arguments if a[2] is not None]
        if len(args) != len(required) or set(kwargs) - {a[0] for a in optional}:
            form = ", ".join([a[0] for a in required] + [f"{a[0]}=..." for a in optional])
            raise ValueError(f"expression {text!r}: write {name}({form})")
        given = dict(zip((a[0] for a in required), args), **kwargs)
        opt = {a: (int if typ == "material" else float)(given.get(a, default)) for a, typ, default in q.arguments}
    return (name, opt) if role == "objective" else (name, opt, float(bound))
