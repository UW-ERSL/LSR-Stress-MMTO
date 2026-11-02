"""
display.py
==========
Tables that render as HTML in a notebook and as aligned text in a terminal.

Table(headers, rows, title, note)   one table
Tables(*tables)                     several, shown together
Table(..., scroll=N)                fixed-height, scrollable, for long lists

Nothing here imports IPython; the notebook picks up _repr_html_ on its own.
"""
from __future__ import annotations

import html
import re

# Greek names to the character, so tables read like the paper
_GREEK = {
    "Gamma": "\u0393", "Phi": "\u03a6", "Sigma": "\u03a3", "Delta": "\u0394",
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "eta": "\u03b7",
    "rho": "\u03c1", "sigma": "\u03c3", "tau": "\u03c4", "phi": "\u03c6",
}
_GREEK_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(_GREEK, key=len, reverse=True))
                       + r")(?![A-Za-z])")
# a subscript only when the base is a standalone single letter, so r_rho and
# Phi_p convert but Youngs_Modulus does not
_SUB_RE = re.compile(r"(?<![A-Za-z\u0370-\u03ff])([A-Za-z\u0370-\u03ff])"
                     r"_\{?([A-Za-z0-9\u0370-\u03ff]{1,3})\}?")


def symbols_text(s: str) -> str:
    """r_rho -> r_\u03c1,  Gamma -> \u0393,  Phi_p -> \u03a6_p"""
    return _GREEK_RE.sub(lambda m: _GREEK[m.group(1)], s)


def symbols_html(s: str) -> str:
    """As symbols_text, with the subscript as <sub>."""
    out = html.escape(symbols_text(s))
    return _SUB_RE.sub(lambda m: f"{m.group(1)}<sub>{m.group(2)}</sub>", out)

_CSS = (
    "border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,"
    "'Segoe UI',Roboto,sans-serif;font-size:13px;margin:0 0 4px 0;"
)
_TH = ("padding:5px 12px;text-align:{align};border-bottom:2px solid #444;"
       "font-weight:600;white-space:nowrap;")
_TD = ("padding:4px 12px;text-align:{align};border-bottom:1px solid #eee;"
       "white-space:nowrap;")


class Raw:
    """A cell with its own HTML, e.g. a colour swatch, and a text fallback."""

    def __init__(self, html_text, text=""):
        self.html = html_text
        self.text = text

    def __str__(self):
        return self.text


def swatch(colour, text=""):
    """A filled square in the table, for showing a material's colour."""
    return Raw(f"<span style='display:inline-block;width:16px;height:16px;"
               f"background:{colour};border:1px solid #888;border-radius:3px;"
               f"vertical-align:middle;'></span>", text)


def _is_number(value) -> bool:
    try:
        float(str(value).replace("%", "").replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


class Table:
    """Rows of strings, with optional headers, title and footnote."""

    def __init__(self, rows, headers=None, title=None, note=None,
                 scroll=None, highlight=None, symbol_cols=(0,)):
        self.rows = [[c if isinstance(c, Raw) else ("" if c is None else str(c))
                      for c in r] for r in rows]
        self.headers = [str(h) for h in headers] if headers else None
        self.title = title
        self.note = note
        self.scroll = scroll          # max height in px; None for no scrolling
        self.highlight = highlight or (lambda row: False)
        self.symbol_cols = set(symbol_cols or ())

    def _aligns(self):
        ncol = max([len(r) for r in self.rows] + [len(self.headers or [])])
        out = []
        for c in range(ncol):
            col = [r[c] for r in self.rows
                   if c < len(r) and not isinstance(r[c], Raw) and r[c] != ""]
            out.append("right" if col and all(_is_number(v) for v in col) else "left")
        return out

    def _text(self, value, col=None):
        if col is None or col in self.symbol_cols:
            return symbols_text(value)
        return value

    def __repr__(self):
        ncol = max([len(r) for r in self.rows] + [len(self.headers or [])])
        widths = [0] * ncol
        for r in [self.headers] if self.headers else []:
            for c, v in enumerate(r):
                widths[c] = max(widths[c], len(v))
        for r in self.rows:
            for c, v in enumerate(r):
                widths[c] = max(widths[c], len(str(v)))

        lines = []
        if self.title:
            lines.append(symbols_text(self.title))
        if self.headers:
            lines.append("  ".join(symbols_text(h).ljust(widths[i])
                                   for i, h in enumerate(self.headers)))
            lines.append("  ".join("-" * widths[i] for i in range(ncol)))
        for r in self.rows:
            lines.append("  ".join(
                self._text(str(r[i]) if i < len(r) else "", i).ljust(widths[i])
                for i in range(ncol)))
        if self.note:
            lines.append(symbols_text(self.note))
        return "\n".join(lines)

    def _repr_html_(self):
        aligns = self._aligns()
        parts = []
        if self.title:
            parts.append(f"<div style='font-weight:600;font-size:14px;"
                         f"margin:10px 0 6px 0;'>{symbols_html(self.title)}</div>")

        body = [f"<table style=\"{_CSS}\">"]
        if self.headers:
            body.append("<thead><tr>")
            for i, h in enumerate(self.headers):
                a = aligns[i] if i < len(aligns) else "left"
                body.append(f"<th style=\"{_TH.format(align=a)}\">"
                            f"{symbols_html(h)}</th>")
            body.append("</tr></thead>")
        body.append("<tbody>")
        for r in self.rows:
            bold = ";font-weight:600" if self.highlight(r) else ""
            body.append("<tr>")
            for i in range(len(aligns)):
                a = aligns[i]
                cell = r[i] if i < len(r) else ""
                if isinstance(cell, Raw):
                    v = cell.html
                elif i in self.symbol_cols:
                    v = symbols_html(cell)
                else:
                    v = html.escape(cell)
                body.append(f"<td style=\"{_TD.format(align=a)}{bold}\">{v}</td>")
            body.append("</tr>")
        body.append("</tbody></table>")
        table = "".join(body)

        if self.scroll:
            table = (f"<div style='max-height:{self.scroll}px;overflow-y:auto;"
                     f"border:1px solid #ddd;border-radius:4px;padding:2px 6px;'>"
                     f"{table}</div>")
        parts.append(table)
        if self.note:
            parts.append(f"<div style='font-size:12px;color:#666;margin-top:4px;'>"
                         f"{symbols_html(self.note)}</div>")
        return "".join(parts)


class Note:
    """One highlighted sentence, for a result that should not be missed."""

    def __init__(self, text, tone="info"):
        self.text = text
        self.tone = tone

    def __repr__(self):
        return symbols_text(self.text)

    def _repr_html_(self):
        edge = {"info": "#3b7dd8", "warn": "#c77700"}.get(self.tone, "#3b7dd8")
        back = {"info": "#f2f7fd", "warn": "#fdf6ec"}.get(self.tone, "#f2f7fd")
        return (f"<div style=\"border-left:4px solid {edge};background:{back};"
                f"padding:9px 14px;margin:8px 0;border-radius:3px;"
                f"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
                f"Roboto,sans-serif;font-size:13.5px;color:#222;line-height:1.5;\">"
                f"{symbols_html(self.text)}</div>")


class Tables:
    """Several tables shown one after another."""

    def __init__(self, *tables):
        self.tables = [t for t in tables if t is not None]

    def __iter__(self):
        return iter(self.tables)

    def __repr__(self):
        return "\n\n".join(repr(t) for t in self.tables)

    def _repr_html_(self):
        return "".join(t._repr_html_() for t in self.tables)
