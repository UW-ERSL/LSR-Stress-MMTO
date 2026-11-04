"""
dark_and_light_3d_view.py
=========================
The 3D view light (white, the default) or dark, and every drawing in it that is not a material colour
made readable on it: a dark line, point or text is drawn light in the dark view. Material colours, and
any drawing coloured by values (a density, a stress), are left as they are.

The window's own drawings take their colour for the current view when they are drawn:

    colour = colour_for("#222222", dark)          the colour to draw with
    mark(actor.GetProperty(), "#222222", dark)    and what it is in the light view, for a later switch

A drawing may also carry its own colour for the dark view (dark_colour="#ffffff"), in place of the
lightened one.

Drawings made elsewhere (the part, its bounding box, the axes, PyTO-AD's highlights) are themed by
apply(), which install() runs at the start of every frame, so none shows in the wrong colour first.
"""
import vtk

DARK_BACKGROUND, DARK_TOP = "#161d2b", "#3b4762"
LIGHT_TEXT = "#d7dce3"
_light = {}               # colour property's address -> its colour in the light view
_set = {}                 # colour property's address -> (dark?, the colour set here), to see it is still ours
_dark = {}                # colour property's address -> its own colour in the dark view, where it has one


def _unit(c):
    """Some drawings give colours as 0 to 255 instead of 0 to 1."""
    return tuple(x / 255 for x in c) if max(c) > 1 else tuple(c)


def _rgb(colour):
    if isinstance(colour, str):
        import pyvista as pv
        return tuple(pv.Color(colour).float_rgb)
    return _unit(tuple(colour))


def _lum(c):
    c = _unit(c)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def colour_for(colour, dark, text=False, dark_colour=None):
    """The colour a drawing of this light-view colour is drawn with: unchanged in the light view; in the
    dark view its own dark_colour when it has one, else a dark line or point lightened, a dark text made
    light grey."""
    c = _rgb(colour)
    if dark and dark_colour is not None:
        return _rgb(dark_colour)
    if not dark or _lum(c) >= (0.6 if text else 0.2):
        return c
    return _rgb(LIGHT_TEXT) if text else tuple(x + (1 - x) * 0.75 for x in c)


def _key(prop):
    return prop.GetAddressAsString(prop.GetClassName())


def _note(prop, light, dark):
    key = _key(prop)
    _light[key] = light
    _set[key] = (dark, tuple(round(x, 6) for x in prop.GetColor()))


def mark(prop, colour, dark, text=False, dark_colour=None):
    """A drawing made with colour_for(colour, dark, text, dark_colour): its colour set for the view, and its
    colours kept, so a switch of the view recolours it and apply() leaves it alone meanwhile."""
    prop.SetColor(*colour_for(colour, dark, text, dark_colour))
    if dark_colour is None:
        _dark.pop(_key(prop), None)
    else:
        _dark[_key(prop)] = dark_colour
    _note(prop, _rgb(colour), dark)


def _theme(prop, dark, text):
    """A colour property of a drawing made elsewhere: its colour as first seen is its light-view colour.
    One whose colour is no longer the one set here is a new drawing (VTK may reuse the address of a
    deleted one), so its colour is taken again."""
    key = _key(prop)
    now = tuple(round(x, 6) for x in prop.GetColor())
    if key in _set and _set[key][1] == now:
        if _set[key][0] == dark:
            return
    else:
        _light[key] = prop.GetColor()
        _dark.pop(key, None)
    prop.SetColor(*colour_for(_light[key], dark, text, _dark.get(key)))
    _note(prop, _light[key], dark)


def _text_props(actor):
    props = []
    for getter in ("GetTextProperty", "GetLabelTextProperty", "GetTitleTextProperty"):
        f = getattr(actor, getter, None)
        if f is not None:
            try:
                p = f()
                if p is not None:
                    props.append(p)
            except TypeError:
                pass
    mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
    f = getattr(mapper, "GetLabelTextProperty", None)
    if f is not None:
        props.append(f())
    try:                                                # point labels: the text is on the label hierarchy
        producer = mapper.GetInputConnection(0, 0).GetProducer()
        f = getattr(producer, "GetTextProperty", None)
        if f is not None and f() is not None:
            props.append(f())
    except Exception:  # noqa: BLE001
        pass
    return props


def _by_values(actor):
    """True for a drawing coloured by values or materials, which keeps its colours in both views."""
    m = actor.GetMapper()
    data = m.GetInput() if m is not None else None
    return bool(m is not None and m.GetScalarVisibility() and data is not None and
                (data.GetPointData().GetScalars() is not None or data.GetCellData().GetScalars() is not None))


def set_background(plotter, dark):
    if dark:
        plotter.set_background(DARK_BACKGROUND, top=DARK_TOP)
    else:
        plotter.set_background("white")
        plotter.renderer.GradientBackgroundOff()


def apply(plotter, dark):
    """Every drawing in the view themed for it (those already themed for it are skipped quickly)."""
    ren = plotter.renderer
    actors = ren.GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        a = actors.GetNextActor()
        if not _by_values(a):
            _theme(a.GetProperty(), dark, text=False)
    a2 = ren.GetActors2D()
    a2.InitTraversal()
    for _ in range(a2.GetNumberOfItems()):
        a = a2.GetNextActor2D()
        for p in _text_props(a):
            _theme(p, dark, text=True)
        if isinstance(a, vtk.vtkAxisActor2D):           # a ruler's line and ticks
            _theme(a.GetProperty(), dark, text=False)
    for bar in getattr(plotter, "scalar_bars", {}).values():
        for p in (bar.GetLabelTextProperty(), bar.GetTitleTextProperty()):
            _theme(p, dark, text=True)
    props = ren.GetViewProps()
    props.InitTraversal()
    for _ in range(props.GetNumberOfItems()):
        prop = props.GetNextProp()
        if isinstance(prop, vtk.vtkCubeAxesActor):
            for i in range(3):
                for p in (prop.GetTitleTextProperty(i), prop.GetLabelTextProperty(i)):
                    _theme(p, dark, text=True)


def install(plotter, is_dark):
    """apply() at the start of every frame: is_dark() gives the view's current mode."""
    plotter.renderer.AddObserver("StartEvent", lambda *_: apply(plotter, is_dark()))
