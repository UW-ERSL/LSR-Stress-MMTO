"""
test_database_versions.py
=========================
The versions of a material database (materials/database_versions.py): numbered v1, v2, ... in a folder of the
database's own beside it, each with the network trained on it; a state no version has kept (record), the one in
use found by its contents, put back in the database's place, a note given, deleted (never the one in use, and
no number given twice), renamed with the database, copied with it, and the dated versions of earlier releases
numbered.

    python -m pytest tests/materials/test_database_versions.py
"""
import os
import shutil
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DATABASE = os.path.join(ROOT, "data", "databases", "6Materials_AF_3Attr")


def _own(folder):
    """The 6-material database and its network, copied into folder/materials as steels; its path."""
    os.makedirs(os.path.join(folder, "materials"))
    xlsx = os.path.join(folder, "materials", "steels.xlsx")
    shutil.copy(DATABASE + ".xlsx", xlsx)
    shutil.copy(DATABASE + ".nt", os.path.join(folder, "materials", "steels.nt"))
    return xlsx


def _rows(xlsx):
    from lsr_mmto.materials.databases import database_table
    return database_table(xlsx)[0]


def _two_materials(xlsx):
    """The database cut down to its first two materials, written in place (as Excel would)."""
    from lsr_mmto.materials.databases import database_table, material_display, write_database
    names, attributes, units, values = database_table(xlsx)
    colours, labels = material_display(xlsx)
    write_database(xlsx, attributes, units, names[:2], labels[:2], colours[:2], values[:2].tolist())


def _recoloured(xlsx):
    """Only the first material's colour changed, written in place."""
    from lsr_mmto.materials.databases import database_table, material_display, write_database
    names, attributes, units, values = database_table(xlsx)
    colours, labels = material_display(xlsx)
    write_database(xlsx, attributes, units, names, labels, ["#123456"] + list(colours[1:]), values.tolist())


def test_kept_numbered_in_use_restored_and_deleted():
    """The first record keeps v1 with its network; the same state again keeps nothing; a change made outside
    (Excel) is kept as v2, without the network trained on other numbers; the one in use is found by its contents;
    restoring v1 puts back its six materials and its network and makes it the one in use, every version staying;
    the version in use can't be deleted; a deleted number is not given again; the folder's list of databases does
    not show the versions."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        assert V.versions(xlsx) == [] and V.in_use(xlsx) is None
        v1 = V.record(xlsx)
        assert (v1.number, v1.name, v1.materials) == (1, "v1", 6) and os.path.isfile(v1.network)
        assert V.record(xlsx) == v1 and len(V.versions(xlsx)) == 1     # nothing new: nothing kept
        _two_materials(xlsx)
        v2 = V.record(xlsx)
        assert (v2.number, v2.materials) == (2, 2) and not os.path.isfile(v2.network)
        assert [v.number for v in V.versions(xlsx)] == [2, 1] and V.in_use(xlsx) == v2
        assert sorted(os.listdir(os.path.join(d, "materials"))) == ["steels.nt", "steels.xlsx", "versions"]
        V.restore(v1, xlsx)
        assert len(_rows(xlsx)) == 6 and V.in_use(xlsx) == v1 and len(V.versions(xlsx)) == 2
        assert os.path.isfile(os.path.splitext(xlsx)[0] + ".nt")
        with pytest.raises(ValueError, match="in use"):
            V.delete(v1, xlsx)
        V.delete(v2, xlsx)
        assert not os.path.exists(v2.path) and [v.number for v in V.versions(xlsx)] == [1]
        _two_materials(xlsx)
        assert V.record(xlsx).number == 3                              # 2 is never given again


def test_a_colour_alone_is_a_new_version_and_a_network_trained_on_other_numbers_is_not_kept():
    """Every change, a colour too, is a version of its own; the network beside the database goes with a version
    unless its record says it was trained on other numbers, and a restored version with no network sets aside one
    that does not fit it."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        V.record(xlsx)
        _recoloured(xlsx)
        v2 = V.record(xlsx)
        assert v2.number == 2 and os.path.isfile(v2.network)           # only a colour: the network still goes
        _two_materials(xlsx)
        v3 = V.record(xlsx)
        assert not os.path.isfile(v3.network)                          # other numbers: not kept with it
        V.restore(v3, xlsx)                                             # in use already: nothing set aside
        V.restore(v2, xlsx)
        V.restore(v3, xlsx)
        assert not os.path.isfile(os.path.splitext(xlsx)[0] + ".nt")
        assert os.path.isfile(os.path.splitext(xlsx)[0] + ".previous.nt")


def test_a_note_a_version_from_another_file_and_an_included_database():
    """A note is kept in the version's name and can be taken away, its number staying; keep with source keeps
    that file (the included database a copy was made from) as the database's version; a database that comes
    with LSR MMTO gets no versions."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        _two_materials(xlsx)
        v1 = V.keep(xlsx, source=DATABASE + ".xlsx")
        assert v1.materials == 6 and os.path.dirname(v1.path) == os.path.join(d, "materials", "versions", "steels")
        noted = V.set_note(v1, "as it came")
        assert noted.name == "v1 as it came" and os.path.isfile(noted.network) and not os.path.exists(v1.path)
        with pytest.raises(ValueError):
            V.set_note(noted, "a/b")
        assert V.set_note(noted, "").name == "v1"
    assert V.record(DATABASE + ".xlsx") is None
    assert not os.path.isdir(os.path.join(os.path.dirname(DATABASE), "versions", "6Materials_AF_3Attr"))


def test_renamed_with_its_network_and_versions_and_a_copy_takes_them_along():
    """rename_database renames the database, its network and its versions together, and refuses a name another
    database has or had; copy_versions gives a copy (Save As) the versions and the last number given."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as e:
        xlsx = _own(d)
        V.record(xlsx)
        _two_materials(xlsx)
        V.record(xlsx)
        shutil.copy(DATABASE + ".xlsx", os.path.join(d, "materials", "other.xlsx"))
        with pytest.raises(ValueError, match="already"):
            V.rename_database(xlsx, "other")
        new = V.rename_database(xlsx, "alloys")
        assert sorted(os.listdir(os.path.join(d, "materials"))) == ["alloys.nt", "alloys.xlsx", "other.xlsx",
                                                                       "versions"]
        assert [v.number for v in V.versions(new)] == [2, 1] and V.in_use(new).number == 2
        os.makedirs(os.path.join(e, "materials"))
        copy = os.path.join(e, "materials", "alloys.xlsx")
        shutil.copy(new, copy)
        V.copy_versions(new, copy)
        V.copy_versions(new, copy)                                      # again: nothing doubled
        assert [v.number for v in V.versions(copy)] == [2, 1] and V.in_use(copy).number == 2
        V.delete(V.versions(copy)[1], copy)
        _recoloured(copy)
        assert V.record(copy).number == 3


def test_the_dated_versions_of_earlier_releases_are_numbered():
    """Versions kept by an earlier LSR MMTO (versions/steels <date>[ first].xlsx) are moved into the database's
    folder and numbered oldest first, with their networks."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        old = os.path.join(d, "materials", "versions")
        os.makedirs(old)
        shutil.copy(DATABASE + ".xlsx", os.path.join(old, "steels 2026-11-10 09-15-40 first.xlsx"))
        shutil.copy(DATABASE + ".nt", os.path.join(old, "steels 2026-11-10 09-15-40 first.nt"))
        shutil.copy(DATABASE + ".xlsx", os.path.join(old, "steels 2026-11-15 06-48-02.xlsx"))
        kept = V.versions(xlsx)
        assert [v.number for v in kept] == [2, 1] and os.path.isfile(kept[1].network)
        assert sorted(os.listdir(old)) == ["steels"]
        _two_materials(xlsx)
        assert V.record(xlsx).number == 3


def test_nothing_is_lost_by_a_restore_a_retraining_a_broken_file_or_a_failed_rename():
    """A restore first keeps the database as it is now when no version has it (a change made in Excel while
    Versions was open); a network trained again on the same contents replaces the older one kept with the
    version in use, so a restore never brings back an older network over a newer; a database that cannot be read
    keeps no version and breaks nothing; a rename that fails part way is undone, the database staying with its
    network and versions; a network that fits no version is set aside under a name of its own."""
    from unittest import mock
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        nt = os.path.splitext(xlsx)[0] + ".nt"
        v1 = V.record(xlsx)
        _two_materials(xlsx)                                            # in Excel, not seen yet
        V.restore(v1, xlsx)
        assert [(v.number, v.materials) for v in V.versions(xlsx)] == [(2, 2), (1, 6)]
        with open(nt, "ab") as f:
            f.write(b" ")                                               # trained again: other bytes, same record
        assert V.record(xlsx) == V.in_use(xlsx) and open(v1.network, "rb").read() == open(nt, "rb").read()
        open(xlsx, "wb").write(b"not a workbook")
        assert V.record(xlsx) is None and V.in_use(xlsx) is None and len(V.versions(xlsx)) == 2
        V.restore(v1, xlsx)
        real, calls = os.replace, []

        def fails_second(a, b):
            calls.append(a)
            if len(calls) == 2:
                raise PermissionError("in use")
            real(a, b)
        with mock.patch("os.replace", fails_second), pytest.raises(PermissionError):
            V.rename_database(xlsx, "alloys")
        assert os.path.isfile(xlsx) and os.path.isfile(nt) and len(V.versions(xlsx)) == 2
        assert not os.path.exists(os.path.join(d, "materials", "alloys.xlsx"))
        open(os.path.splitext(xlsx)[0] + ".previous.nt", "wb").write(b"older")
        _two_materials(xlsx)
        V.restore(V.versions(xlsx)[1], xlsx)                            # its network goes back with it
        V.restore(V.versions(xlsx)[0], xlsx)                            # v3 has none: the one there set aside
        assert open(os.path.splitext(xlsx)[0] + ".previous.nt", "rb").read() == b"older"
        assert os.path.isfile(os.path.splitext(xlsx)[0] + ".previous 2.nt")
