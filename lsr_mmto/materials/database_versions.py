"""
database_versions.py
====================
The versions of a material database: every state of it that a project used is kept, numbered v1, v2, ...,
so any of them can be brought back. They live in a folder of their own beside the database, out of the
Materials window's list of databases:

    materials/steels.xlsx, steels.nt                      the database and its trained network
    materials/versions/steels/v1.xlsx, v1.nt              the database as it first was, with its network
    materials/versions/steels/v2 softer.xlsx              a later version, with a note of the user's
    materials/versions/steels/numbers.json                the last number given, so none is given twice

A version is kept whenever the database's contents (every cell, the colours and labels) are ones no version
has: when the database is saved in the editor, imported, converted, chosen, and when a change made outside the
app (in Excel) is found: as a project opens, as Materials opens, and as a run starts (record). The version in
use is the one whose contents match the database. A version keeps the network trained on its contents (never
one whose record says its numbers differ), so a restored version needs no new training. A database that comes with LSR MMTO has no versions: it is never
changed. A database copied into another project folder (Save As) takes its versions with it.

    versions(xlsx)                   the versions of a database, newest first
    in_use(xlsx, among)              the version whose contents match the database, or None
    record(xlsx)                     a version kept when the database is one no version has; its network kept
                                     with the version in use when it was trained on it
    keep(xlsx, source)               a version kept: the database now, or the file source
    restore(version, xlsx)           a version back in the database's place, with its network
    delete(version, xlsx)            a version and its network deleted (never the one in use)
    set_note(version, note)          a version's note changed
    rename_database(xlsx, name)      a database renamed with its network and all its versions
    copy_versions(xlsx, to)          the versions of one database copied as the versions of another
    name_taken(folder, name)         a database of that name, or its versions, already in the folder
"""
import functools
import hashlib
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass
from typing import Any

FOLDER = "versions"
NUMBERS = "numbers.json"
NOT_IN_A_NAME = '<>:"/\\|?*'                    # characters a file name on Windows cannot have
_OLD = r" (\d{4}-\d\d-\d\d \d\d-\d\d-\d\d)( first)?\.xlsx"      # the dated versions kept before numbers
_LOCK = threading.RLock()          # one change of versions at a time: a run's thread keeps them too


def _one_at_a_time(fn):
    """fn run while no other change of versions is going (a run's thread and the window at once)."""
    @functools.wraps(fn)
    def locked(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return locked


@dataclass(frozen=True)
class Version:
    """A kept version: its .xlsx, its number, the user's note ("" for none), when its contents were saved
    (seconds since the epoch) and how many materials it has (-1 when its file cannot be read)."""
    path: str
    number: int
    note: str
    when: float
    materials: int

    @property
    def network(self) -> str:
        """Its network's file (there only when the version has one)."""
        return os.path.splitext(self.path)[0] + ".nt"

    @property
    def name(self) -> str:
        """"v3", or "v3 softer" with its note."""
        return f"v{self.number}" + (f" {self.note}" if self.note else "")


def _stem(xlsx: str) -> str:
    return os.path.splitext(os.path.basename(xlsx))[0]


def _folder(xlsx: str) -> str:
    """The folder of the database's versions."""
    return os.path.join(os.path.dirname(os.path.abspath(xlsx)), FOLDER, _stem(xlsx))


def _last(folder: str) -> int:
    """The last number given in folder (0 for none)."""
    try:
        with open(os.path.join(folder, NUMBERS), encoding="utf-8") as f:
            return int(json.load(f).get("last", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def _set_last(folder: str, last: int) -> None:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, NUMBERS), "w", encoding="utf-8") as f:
        json.dump({"last": last}, f)


def _next(folder: str) -> int:
    """The next number to give in folder: after the last one given and after every file there."""
    numbers = [int(m.group(1)) for name in (os.listdir(folder) if os.path.isdir(folder) else [])
               for m in [re.fullmatch(r"v(\d+)(?: .+)?\.xlsx", name)] if m]
    return max([_last(folder)] + numbers) + 1


def _move_with_network(path: str, dest: str) -> None:
    os.replace(path, dest)
    if os.path.isfile(os.path.splitext(path)[0] + ".nt"):
        os.replace(os.path.splitext(path)[0] + ".nt", os.path.splitext(dest)[0] + ".nt")


def _number_old_versions(xlsx: str) -> None:
    """The dated versions an earlier LSR MMTO kept (versions/<name> <date>[ first].xlsx) moved into the
    database's own folder and numbered, oldest first, after any numbers already given."""
    parent = os.path.join(os.path.dirname(os.path.abspath(xlsx)), FOLDER)
    if not os.path.isdir(parent):
        return
    pattern = re.compile(re.escape(_stem(xlsx)) + _OLD)
    old = [(m.group(1), not m.group(2), name) for name in os.listdir(parent) for m in [pattern.fullmatch(name)] if m]
    if not old:
        return
    folder = _folder(xlsx)
    last = _next(folder) - 1
    for _, _, name in sorted(old):                              # by date, the first one first
        last += 1
        os.makedirs(folder, exist_ok=True)
        _move_with_network(os.path.join(parent, name), os.path.join(folder, f"v{last}.xlsx"))
    _set_last(folder, last)


def _key(path: str) -> str:
    """A file's contents, as a hash of its bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


_COUNTS: dict[str, int] = {}                      # a file's bytes (their hash): its count of materials


def _count(path: str) -> int:
    """How many materials the file has, -1 when it cannot be read; read once for each content."""
    from lsr_mmto.materials.databases import database_table
    try:
        key = _key(path)
    except OSError:
        return -1
    if key not in _COUNTS:
        try:
            _COUNTS[key] = len(database_table(path)[0])
        except Exception:                                               # noqa: BLE001  (it cannot be read)
            _COUNTS[key] = -1
    return _COUNTS[key]


def _version(path: str) -> Version | None:
    m = re.fullmatch(r"v(\d+)(?: (.+))?\.xlsx", os.path.basename(path))
    if m is None or not os.path.isfile(path):
        return None
    return Version(path, int(m.group(1)), m.group(2) or "", os.path.getmtime(path), _count(path))


def _read(folder: str) -> list[Version]:
    """The versions in folder, newest first; a file that is not one is left out."""
    if not os.path.isdir(folder):
        return []
    found = [_version(os.path.join(folder, name)) for name in os.listdir(folder)]
    return sorted((v for v in found if v is not None), key=lambda v: v.number, reverse=True)


@_one_at_a_time
def versions(xlsx: str) -> list[Version]:
    """The kept versions of the database xlsx, newest first."""
    _number_old_versions(xlsx)
    return _read(_folder(xlsx))


_STAMPS: dict[str, Any] = {}                     # a file's bytes (their hash): its contents, read once


def _contents(path: str) -> Any:
    """What a version is told apart by: every cell of the database's first sheet, and its colours and labels
    (network_stamp.database_stamp), read once for each content of a file (known by a hash of its bytes, so a
    file rewritten within the same second is read again)."""
    from lsr_mmto.materials.network_stamp import database_stamp
    key = _key(path)
    if key not in _STAMPS:
        _STAMPS[key] = database_stamp(path)
    return _STAMPS[key]


def in_use(xlsx: str, among: list[Version] | None = None) -> Version | None:
    """The version whose contents are the database's now (the newest such), or None: among, or its versions.
    None too when the database cannot be read (a file Excel left broken, say)."""
    try:
        now = _contents(xlsx)
    except Exception:                                                   # noqa: BLE001  (not there, or unreadable)
        return None
    for v in versions(xlsx) if among is None else among:
        try:
            if _contents(v.path) == now:
                return v
        except Exception:                                               # noqa: BLE001  (it cannot be read)
            continue
    return None


def _network_fits(nt: str, xlsx: str) -> bool:
    """The network nt may go with the contents of xlsx: its record of the database it was trained on does not
    say the numbers differ (network_stamp: anything but CHANGED). A run still checks it as for any network,
    asking first when only colours, labels or units differ or it has no record."""
    from lsr_mmto.materials.network_stamp import CHANGED, compare
    try:
        return os.path.isfile(nt) and compare(xlsx, nt).state != CHANGED
    except Exception:                                                   # noqa: BLE001  (not a network)
        return False


@_one_at_a_time
def keep(xlsx: str, source: str | None = None) -> Version:
    """A version of the database xlsx kept, numbered after the last number given: the file source as the
    version (an included database the database was copied from, say), else the database as it is now. Its
    network goes with it unless its record says it was trained on other numbers (_network_fits)."""
    source = xlsx if source is None else source
    _number_old_versions(xlsx)
    folder = _folder(xlsx)
    number = _next(folder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"v{number}.xlsx")
    _set_last(folder, number)                                       # first: a number is never given twice
    shutil.copy2(source, path)
    network = os.path.splitext(source)[0] + ".nt"
    if _network_fits(network, path):
        shutil.copy2(network, os.path.splitext(path)[0] + ".nt")
    return _version(path)


@_one_at_a_time
def record(xlsx: str) -> Version | None:
    """The database's state kept as a version when no version has it (a change saved in the app, or made
    outside it in Excel); and the network beside the database kept with the version in use when it goes with
    its contents (_network_fits) and differs from the one the version has (a network trained again replaces the
    older one there too). The version in use, or None for a database that comes with LSR MMTO, is not there,
    or cannot be read now (open in Excel while it saves, say: the next check keeps it)."""
    from lsr_mmto.materials.databases import in_shipped_folder
    if not os.path.isfile(xlsx) or in_shipped_folder(xlsx):
        return None
    try:
        _contents(xlsx)                                 # a file that cannot be read is not kept: kept once read
        version = in_use(xlsx) or keep(xlsx)
        network = os.path.splitext(xlsx)[0] + ".nt"
        if _network_fits(network, version.path) and \
                (not os.path.isfile(version.network) or _key(network) != _key(version.network)):
            shutil.copy2(network, version.network)
        return version
    except Exception:                                                   # noqa: BLE001  (read or written later)
        return None


def _set_aside(network: str, xlsx: str) -> None:
    """A network that does not fit the database moved to <name>.previous.nt, or .previous 2.nt, ... when that
    is taken: no network set aside before is written over."""
    base, k = os.path.splitext(xlsx)[0] + ".previous", 2
    dest = base + ".nt"
    while os.path.exists(dest):
        dest, k = f"{base} {k}.nt", k + 1
    os.replace(network, dest)


@_one_at_a_time
def restore(version: Version, xlsx: str) -> None:
    """The version back as the database xlsx, with its network. The database as it is now is kept first, with
    its network, when no version has them (record), so nothing is lost; a network beside it that goes with no
    version is set aside as <name>.previous.nt. ValueError for a version that cannot be read."""
    if version.materials < 0:
        raise ValueError(f"{version.name} cannot be read")
    record(xlsx)
    network = os.path.splitext(xlsx)[0] + ".nt"
    if os.path.isfile(network) and not _network_fits(network, xlsx):
        _set_aside(network, xlsx)
    shutil.copy2(version.path, xlsx)
    if os.path.isfile(version.network):
        shutil.copy2(version.network, network)
    elif os.path.isfile(network) and not _network_fits(network, xlsx):
        _set_aside(network, xlsx)


@_one_at_a_time
def delete(version: Version, xlsx: str) -> None:
    """The version and its network deleted; its number is not given again. ValueError for the version in use:
    it is the database as it is now."""
    current = in_use(xlsx)
    if current is not None and current.path == version.path:
        raise ValueError(f"{version.name} is the version in use")
    for path in (version.path, version.network):
        if os.path.isfile(path):
            os.remove(path)


def note_error(note: str) -> str | None:
    """Why a note cannot be one, or None."""
    bad = sorted(set(note) & set(NOT_IN_A_NAME))
    if bad:
        return f"a note cannot have {' '.join(bad)}"
    if note != note.strip():
        return "a note cannot start or end with a space"
    return None


@_one_at_a_time
def set_note(version: Version, note: str) -> Version:
    """The version with its note changed ("" for none): its files renamed, its number kept."""
    note = note.strip()
    error = note_error(note)
    if error:
        raise ValueError(error)
    folder = os.path.dirname(version.path)
    dest = os.path.join(folder, f"v{version.number}" + (f" {note}" if note else "") + ".xlsx")
    if dest != version.path:
        if os.path.exists(dest):
            raise ValueError(f"{os.path.basename(dest)} is already there")
        _move_with_network(version.path, dest)
    return _version(dest)


def name_taken(folder: str, name: str) -> bool:
    """A database of that name is in folder, or its versions are, in the numbered or the earlier dated layout
    (a new database never takes another's)."""
    old = re.compile(re.escape(name) + _OLD)
    parent = os.path.join(folder, FOLDER)
    return os.path.exists(os.path.join(folder, name + ".xlsx")) or os.path.exists(os.path.join(parent, name)) or \
        (os.path.isdir(parent) and any(old.fullmatch(f) for f in os.listdir(parent)))


@_one_at_a_time
def rename_database(xlsx: str, name: str) -> str:
    """The database renamed name, in its folder, with its network, the networks set aside, and all its
    versions. The database's file moves first (the one Excel may hold open), and every move is undone when a
    later one fails, so the database never ends up apart from its network and versions. ValueError for a name
    a file cannot have, or one another database has or had. Returns the new path."""
    from lsr_mmto.materials.databases import in_shipped_folder
    name = name.strip()
    if not name or set(name) & set(NOT_IN_A_NAME) or name in (".", ".."):
        raise ValueError("a name can't be empty or have any of " + " ".join(NOT_IN_A_NAME))
    if in_shipped_folder(xlsx):
        raise ValueError("a database that comes with LSR MMTO is not renamed")
    folder = os.path.dirname(os.path.abspath(xlsx))
    if name == _stem(xlsx):
        return xlsx
    if name.lower() != _stem(xlsx).lower() and name_taken(folder, name):   # only the case changed: same files
        raise ValueError(f"a database named {name} is already in the folder, or was")
    _number_old_versions(xlsx)
    dest = os.path.join(folder, name + ".xlsx")
    base, new_base = os.path.splitext(xlsx)[0], os.path.splitext(dest)[0]
    moves = [(xlsx, dest)]
    if os.path.isdir(_folder(xlsx)):
        moves.append((_folder(xlsx), _folder(dest)))
    for f in sorted(os.listdir(folder)):
        if f.startswith(os.path.basename(base) + ".") and f.endswith(".nt"):    # .nt, .previous.nt, .previous 2.nt
            moves.append((os.path.join(folder, f), new_base + f[len(os.path.basename(base)):]))
    done = []
    try:
        for a, b in moves:
            if os.path.normcase(a) == os.path.normcase(b) and a != b:       # only the case: through a free name
                os.replace(a, a + ".renaming")
                a = a + ".renaming"
            os.replace(a, b)
            done.append((a, b))
    except OSError:
        for a, b in reversed(done):
            os.replace(b, a.removesuffix(".renaming"))
        raise
    return dest


@_one_at_a_time
def copy_versions(xlsx: str, to: str) -> None:
    """The versions of the database xlsx, with their networks and the last number given, copied as the
    versions of the database to (a copy of it in another folder); a version to already has is left as it is."""
    source, folder = _folder(xlsx), _folder(to)
    for v in versions(xlsx):
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, os.path.basename(v.path))
        if not any(w.number == v.number for w in _read(folder)):      # a number the copy has: left as it is
            shutil.copy2(v.path, dest)
            if os.path.isfile(v.network):
                shutil.copy2(v.network, os.path.splitext(dest)[0] + ".nt")
    if os.path.isdir(folder):
        _set_last(folder, max(_last(folder), _last(source)))
