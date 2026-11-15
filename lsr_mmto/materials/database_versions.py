"""
database_versions.py
====================
Older versions of a material database, kept so that a change can be undone. They live in a folder
versions beside the database, out of the Materials window's list of databases:

    materials/steels.xlsx, steels.nt                         the database and its trained network
    materials/versions/steels 2026-11-15 06-48-02.xlsx       a version kept before a save
    materials/versions/steels 2026-11-10 09-15-40 first.xlsx the database as it first was
    materials/versions/steels 2026-11-10 09-15-40 first.nt   with its network, when it had one

The first version is always kept: a database made or imported in the editor, the one an included database
was copied from, or, for any other database (one made before versions were kept, one imported as it is), the
database as it was before its first change in the editor or its first conversion of units. Any later version
is kept only when asked for. A version keeps the network beside it, so a restored version needs no new
training. A database copied into another project folder (Save As) takes its versions with it.

    versions(xlsx)                  the versions of a database, newest first
    has_first(xlsx)                 whether its first version is kept
    keep(xlsx, first, source)       a version kept: the database now, or the file source
    restore(version, xlsx)          a version back in the database's place
    delete(version)                 a version and its network deleted
    copy_versions(xlsx, to)         the versions of one database copied as the versions of another
"""
import os
import re
import shutil
import time
from dataclasses import dataclass

FOLDER = "versions"
STAMP = "%Y-%m-%d %H-%M-%S"


@dataclass(frozen=True)
class Version:
    """A kept version: its .xlsx, when it was kept (seconds since the epoch), whether it is the first version,
    and how many materials it has (-1 when its file cannot be read)."""
    path: str
    when: float
    first: bool
    materials: int

    @property
    def network(self) -> str:
        """Its network's file (there only when the version had one)."""
        return os.path.splitext(self.path)[0] + ".nt"


def _folder(xlsx: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(xlsx)), FOLDER)


def _stem(xlsx: str) -> str:
    return os.path.splitext(os.path.basename(xlsx))[0]


def versions(xlsx: str) -> list[Version]:
    """The kept versions of the database xlsx, newest first; a file of the folder that is not one is left out."""
    from lsr_mmto.materials.databases import database_table
    folder, stem = _folder(xlsx), _stem(xlsx)
    if not os.path.isdir(folder):
        return []
    pattern = re.compile(re.escape(stem) + r" (\d{4}-\d\d-\d\d \d\d-\d\d-\d\d)( first)?\.xlsx")
    out = []
    for name in os.listdir(folder):
        m = pattern.fullmatch(name)
        if m is None:
            continue
        path = os.path.join(folder, name)
        try:
            count = len(database_table(path)[0])
        except Exception:                                               # noqa: BLE001
            count = -1                                                  # it cannot be read
        out.append(Version(path, time.mktime(time.strptime(m.group(1), STAMP)), bool(m.group(2)), count))
    return sorted(out, key=lambda v: (v.when, not v.first), reverse=True)


def has_first(xlsx: str) -> bool:
    return any(v.first for v in versions(xlsx))


def keep(xlsx: str, first: bool = False, source: str | None = None) -> Version:
    """A version of the database xlsx kept, with its network when it has one: the file source as the version
    (an included database the database was copied from, say), else the database as it is now."""
    source = xlsx if source is None else source
    folder = _folder(xlsx)
    os.makedirs(folder, exist_ok=True)
    when = time.time()
    path = os.path.join(folder, f"{_stem(xlsx)} {time.strftime(STAMP, time.localtime(when))}"
                                f"{' first' if first else ''}.xlsx")
    k = 1
    while os.path.exists(path):                                         # two in one second: a second later
        path = os.path.join(folder, f"{_stem(xlsx)} {time.strftime(STAMP, time.localtime(when + k))}"
                                    f"{' first' if first else ''}.xlsx")
        k += 1
    shutil.copy2(source, path)
    network = os.path.splitext(source)[0] + ".nt"
    if os.path.isfile(network):
        shutil.copy2(network, os.path.splitext(path)[0] + ".nt")
    return next(v for v in versions(xlsx) if v.path == path)


def restore(version: Version, xlsx: str) -> None:
    """The version back as the database xlsx, with its network; a network of the database now that the version
    did not have is set aside as <name>.previous.nt, since it was trained on other numbers."""
    shutil.copy2(version.path, xlsx)
    network = os.path.splitext(xlsx)[0] + ".nt"
    if os.path.isfile(version.network):
        shutil.copy2(version.network, network)
    elif os.path.isfile(network):
        os.replace(network, os.path.splitext(xlsx)[0] + ".previous.nt")


def delete(version: Version) -> None:
    """The version and its network deleted."""
    for path in (version.path, version.network):
        if os.path.isfile(path):
            os.remove(path)


def copy_versions(xlsx: str, to: str) -> None:
    """The versions of the database xlsx, with their networks, copied as the versions of the database to (a
    copy of it in another folder); a version to already has is left as it is."""
    folder = _folder(to)
    for v in versions(xlsx):
        os.makedirs(folder, exist_ok=True)
        stamp = os.path.basename(v.path)[len(_stem(xlsx)):]            # " 2026-11-15 06-48-02 first.xlsx"
        dest = os.path.join(folder, _stem(to) + stamp)
        if not os.path.exists(dest):
            shutil.copy2(v.path, dest)
            if os.path.isfile(v.network):
                shutil.copy2(v.network, os.path.splitext(dest)[0] + ".nt")
