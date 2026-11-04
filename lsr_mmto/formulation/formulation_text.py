"""
formulation_text.py
===================
The formula text of an objective or a constraint ("Mass()", "MaterialVolumeFraction(2, tau=0.5)")
read into a quantity's name and options: parse_expression, and to_params_form for what
to_params holds.
"""
import re


def parse_expression(text):
    """'Name(a, b=1)' -> ('Name', [a], {'b': 1}). Only a single quantity is accepted. An argument written by
    position may be a word (a material word: 'MassFraction(lightest)'), kept as text; the rest are numbers."""
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
            if eq or not part.isidentifier():
                raise ValueError(f"expression {text!r}: argument {part!r} is not a number") from None
            v = part
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
        opt = {a: _argument(text, typ, words, given.get(a, default)) for a, typ, default, words in q.arguments}
    return (name, opt) if role == "objective" else (name, opt, float(bound))


def _argument(text, typ, words, value):
    """An argument's value as to_params holds it: a whole number for a material (or one of the material words
    it takes, as the word), a number otherwise."""
    if isinstance(value, str):
        if typ == "material" and value in words:
            return value
        allowed = ("a number" if typ != "material" else
                   f"a material number or one of {', '.join(words)}" if words else "a material number")
        raise ValueError(f"expression {text!r}: {value!r} is not {allowed}")
    return int(value) if typ == "material" else float(value)
