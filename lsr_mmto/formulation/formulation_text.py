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
    bound) for a constraint; name is the quantity's, options None or {"material_id", "tau"}."""
    from lsr_mmto.formulation.quantities import DEFAULT_TAU, QUANTITIES
    name, args, kwargs = parse_expression(text)
    by_name = {q.name: q for q in QUANTITIES}
    if name not in by_name:
        raise ValueError(f"expression {text!r}: unknown quantity {name!r}; "
                         f"known: {', '.join(by_name)}")
    q = by_name[name]
    if not (q.objective if role == "objective" else q.constraint):
        raise ValueError(f"{name} cannot be the {role}")
    if name == "MaterialVolumeFraction":
        if len(args) != 1 or set(kwargs) - {"tau"}:
            raise ValueError(f"expression {text!r}: write MaterialVolumeFraction(material, tau=...)")
        opt = {"material_id": int(args[0]), "tau": float(kwargs.get("tau", DEFAULT_TAU))}
    else:
        if args or kwargs:
            raise ValueError(f"expression {text!r}: {name} takes no arguments")
        opt = None
    return (name, opt) if role == "objective" else (name, opt, float(bound))
