"""
test_database_versions.py
=========================
The older versions of a material database (materials/database_versions.py): kept in versions beside it with
their network, listed newest first with the first marked, put back in the database's place, and deleted.

    python -m pytest tests/materials/test_database_versions.py
"""
import os
import shutil
import sys
import tempfile

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
    """The database cut down to its first two materials, written in place."""
    from lsr_mmto.materials.databases import database_table, material_display, write_database
    names, attributes, units, values = database_table(xlsx)
    colours, labels = material_display(xlsx)
    write_database(xlsx, attributes, units, names[:2], labels[:2], colours[:2], values[:2].tolist())


def test_kept_listed_restored_and_deleted():
    """A first version and a later one are kept with the network and listed newest first, the first marked and
    each with its count of materials; the folder's list of databases does not show them; restoring the first
    puts back its six materials and its network; deleting a version deletes its network too."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        assert V.versions(xlsx) == [] and not V.has_first(xlsx)
        first = V.keep(xlsx, first=True)
        assert first.first and first.materials == 6 and os.path.isfile(first.network) and V.has_first(xlsx)
        _two_materials(xlsx)
        os.remove(os.path.splitext(xlsx)[0] + ".nt")                    # the numbers changed: no network
        later = V.keep(xlsx)
        assert not later.first and later.materials == 2 and not os.path.isfile(later.network)
        assert [v.path for v in V.versions(xlsx)] == [later.path, first.path]
        assert sorted(os.listdir(os.path.join(d, "materials"))) == ["steels.xlsx", "versions"]
        V.restore(first, xlsx)
        assert len(_rows(xlsx)) == 6 and os.path.isfile(os.path.splitext(xlsx)[0] + ".nt")
        V.restore(later, xlsx)                                          # a version with no network: the
        assert len(_rows(xlsx)) == 2                                    # network now is set aside
        assert not os.path.isfile(os.path.splitext(xlsx)[0] + ".nt")
        assert os.path.isfile(os.path.splitext(xlsx)[0] + ".previous.nt")
        V.delete(first)
        assert not os.path.exists(first.path) and not os.path.exists(first.network)
        assert [v.path for v in V.versions(xlsx)] == [later.path] and not V.has_first(xlsx)


def test_a_version_from_another_file_and_two_in_one_second():
    """keep with source keeps that file (the included database a copy was made from) under the database's own
    name; two versions kept in the same second get names of their own."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d:
        xlsx = _own(d)
        _two_materials(xlsx)
        first = V.keep(xlsx, first=True, source=DATABASE + ".xlsx")
        assert first.materials == 6 and os.path.basename(first.path).startswith("steels ")
        a, b = V.keep(xlsx), V.keep(xlsx)
        assert a.path != b.path and len(V.versions(xlsx)) == 3


def test_a_copy_takes_its_versions_with_it():
    """copy_versions gives a copy of a database (Save As copies it into another project folder) the original's
    versions, renamed after the copy, with their networks; one the copy already has is left as it is."""
    from lsr_mmto.materials import database_versions as V
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as e:
        xlsx = _own(d)
        first = V.keep(xlsx, first=True)
        os.makedirs(os.path.join(e, "materials"))
        copy = os.path.join(e, "materials", "steels.xlsx")
        shutil.copy(xlsx, copy)
        V.copy_versions(xlsx, copy)
        V.copy_versions(xlsx, copy)                                     # again: nothing doubled
        kept, = V.versions(copy)
        assert kept.first and os.path.basename(kept.path) == os.path.basename(first.path)
        assert os.path.isfile(kept.network) and os.path.dirname(kept.path) == os.path.join(e, "materials",
                                                                                          V.FOLDER)

