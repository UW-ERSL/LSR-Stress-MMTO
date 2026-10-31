"""
database.py
===========
Material databases from the command line (from the repository folder):

    python utils/database.py check my_db.xlsx
        the errors and warnings of a database, and the latent dimension recommended for it

    python utils/database.py template my_db.xlsx Density Youngs_Modulus Yield_Strength
        an empty database to fill in, with a units list under each attribute and a
        Reference sheet; the attributes known are those in ATTRIBUTES, 0-Common/databases.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv):
    sys.path.insert(0, os.path.join(ROOT, "0-Common"))
    from databases import ATTRIBUTES, check_database, recommended_latent_dim, write_template

    if len(argv) >= 2 and argv[0] == "check":
        path = argv[1]
        errors, warnings = check_database(path)
        for e in errors:
            print(f"error:   {e}")
        for w in warnings:
            print(f"warning: {w}")
        if errors:
            print(f"{len(errors)} errors")
            return 1
        d_eff, n_z = recommended_latent_dim(path)
        print(f"no errors. Effective dimension d_eff = {d_eff:.2f}: latent dimension {n_z} recommended")
        return 0
    if len(argv) >= 3 and argv[0] == "template":
        path = write_template(argv[1], argv[2:])
        print(f"wrote {path}")
        return 0
    print(__doc__)
    print("known attributes: " + ", ".join(ATTRIBUTES))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
