import warnings
warnings.filterwarnings("ignore")

import os
import io
import numpy
import pandas

import prody
import yabul

import proteopt
import proteopt.alignment

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def Xtest_atom_rordering():
    # In the below I swapped the position of the second CA
    pdb1_string = """
ATOM      1  N   VAL A   1      -2.900  17.600  15.500  1.00  0.00           N  
ATOM      2  CA  VAL A   1      -3.600  16.400  15.300  1.00  0.00           C  
ATOM      3  C   VAL A   1      -3.000  15.300  16.200  1.00  0.00           C  
ATOM      4  O   VAL A   1      -3.700  14.700  17.000  1.00  0.00           O  
ATOM      5  CB  VAL A   1      -3.500  16.000  13.800  1.00  0.00           C  
ATOM      6  CG1 VAL A   1      -2.100  15.700  13.300  1.00  0.00           C  
ATOM      7  CG2 VAL A   1      -4.600  14.900  13.400  1.00  0.00           C  
ATOM      8  N   LEU A   2      -1.700  15.100  16.000  1.00  0.00           N  
ATOM      9  C   LEU A   2      -1.000  13.900  18.300  1.00  0.00           C  
ATOM     10  O   LEU A   2      -0.900  14.900  19.000  1.00  0.00           O  
ATOM     11  CA  LEU A   2      -0.900  14.100  16.700  1.00  0.00           C  
ATOM     12  CB  LEU A   2       0.600  14.200  16.500  1.00  0.00           C  
ATOM     13  CG  LEU A   2       1.100  14.300  15.100  1.00  0.00           C  
ATOM     14  CD1 LEU A   2       0.400  15.500  14.400  1.00  0.00           C  
ATOM     15  CD2 LEU A   2       2.600  14.400  15.000  1.00  0.00           C  
ATOM     16  N   SER A   3      -1.100  12.600  18.600  1.00  0.00           N  
ATOM     17  CA  SER A   3      -1.100  12.200  20.000  1.00  0.00           C  
ATOM     18  C   SER A   3      -0.100  12.600  21.200  1.00  0.00           C  
ATOM     19  O   SER A   3       1.100  12.800  20.900  1.00  0.00           O  
ATOM     20  CB  SER A   3      -1.100  10.800  20.500  1.00  0.00           C  
ATOM     21  OG  SER A   3       0.200  10.100  20.300  1.00  0.00           O  
ATOM     22  N   GLU A   4      -0.700  12.600  22.400  1.00  0.00           N  
ATOM     23  CA  GLU A   4       0.000  12.900  23.600  1.00  0.00           C  
ATOM     24  C   GLU A   4       1.300  12.100  23.500  1.00  0.00           C  
ATOM     25  O   GLU A   4       2.400  12.600  23.600  1.00  0.00           O  
ATOM     26  CB  GLU A   4      -0.300  12.800  25.100  1.00  0.00           C  
ATOM     27  CG  GLU A   4       0.000  14.000  26.000  1.00  0.00           C  
ATOM     28  CD  GLU A   4       0.300  15.400  25.200  1.00  0.00           C  
ATOM     29  OE2 GLU A   4       1.200  16.000  25.400  1.00  0.00           O  
ATOM     30  OE1 GLU A   4      -0.600  15.500  24.300  1.00  0.00           O  
ATOM     31  N   GLY A   5       1.100  10.800  23.400  1.00  0.00           N  
ATOM     32  CA  GLY A   5       2.200   9.800  23.300  1.00  0.00           C  
ATOM     33  C   GLY A   5       3.200  10.200  22.200  1.00  0.00           C  
ATOM     34  O   GLY A   5       4.400  10.300  22.500  1.00  0.00           O  
ATOM     35  N   GLU A   6       2.700  10.400  21.000  1.00  0.00           N  
ATOM     36  CA  GLU A   6       3.500  10.700  19.800  1.00  0.00           C  
ATOM     37  C   GLU A   6       4.400  11.900  19.900  1.00  0.00           C  
ATOM     38  O   GLU A   6       5.500  11.900  19.400  1.00  0.00           O  
ATOM     39  CB  GLU A   6       2.600  10.700  18.600  1.00  0.00           C  
ATOM     40  CG  GLU A   6       2.000   9.400  18.100  1.00  0.00           C  
ATOM     41  CD  GLU A   6       0.900   9.500  17.000  1.00  0.00           C  
ATOM     42  OE1 GLU A   6       0.700  10.600  16.700  1.00  0.00           O  
ATOM     43  OE2 GLU A   6       0.400   8.500  16.600  1.00  0.00           O  
ATOM     44  N   TRP A   7       3.800  13.000  20.600  1.00  0.00           N  
ATOM     45  CA  TRP A   7       4.500  14.200  20.800  1.00  0.00           C  
ATOM     46  C   TRP A   7       5.700  13.700  21.700  1.00  0.00           C  
ATOM     47  O   TRP A   7       6.900  14.000  21.400  1.00  0.00           O  
ATOM     48  CB  TRP A   7       3.700  15.400  21.300  1.00  0.00           C  
ATOM     49  CG  TRP A   7       2.800  16.100  20.200  1.00  0.00           C  
ATOM     50  CD1 TRP A   7       1.500  16.200  20.100  1.00  0.00           C  
ATOM     51  CD2 TRP A   7       3.300  16.800  19.100  1.00  0.00           C  
ATOM     52  NE1 TRP A   7       1.100  16.900  18.900  1.00  0.00           N  
ATOM     53  CE2 TRP A   7       2.200  17.300  18.300  1.00  0.00           C  
ATOM     54  CE3 TRP A   7       4.600  17.100  18.600  1.00  0.00           C  
ATOM     55  CZ2 TRP A   7       2.300  18.100  17.200  1.00  0.00           C  
ATOM     56  CZ3 TRP A   7       4.700  17.900  17.500  1.00  0.00           C  
ATOM     57  CH2 TRP A   7       3.600  18.400  16.800  1.00  0.00           C  
ATOM     58  N   GLN A   8       5.400  12.900  22.700  1.00  0.00           N  
ATOM     59  CA  GLN A   8       6.300  12.300  23.600  1.00  0.00           C  
ATOM     60  C   GLN A   8       7.600  11.900  22.900  1.00  0.00           C  
ATOM     61  O   GLN A   8       8.700  12.300  23.200  1.00  0.00           O  
ATOM     62  CB  GLN A   8       6.300  12.200  25.100  1.00  0.00           C  
ATOM     63  CG  GLN A   8       7.600  12.200  25.700  1.00  0.00           C  
ATOM     64  CD  GLN A   8       7.700  12.200  27.200  1.00  0.00           C  
ATOM     65  OE1 GLN A   8       8.800  11.900  27.800  1.00  0.00           O  
ATOM     66  NE2 GLN A   8       6.600  12.500  27.800  1.00  0.00           N  
ATOM     67  N   LEU A   9       7.400  11.000  21.900  1.00  0.00           N  
ATOM     68  CA  LEU A   9       8.400  10.400  21.100  1.00  0.00           C  
ATOM     69  C   LEU A   9       9.100  11.400  20.300  1.00  0.00           C  
ATOM     70  O   LEU A   9      10.400  11.500  20.200  1.00  0.00           O  
ATOM     71  CB  LEU A   9       7.900   9.600  19.900  1.00  0.00           C  
ATOM     72  CG  LEU A   9       7.900   8.100  19.900  1.00  0.00           C  
ATOM     73  CD1 LEU A   9       8.200   7.700  21.400  1.00  0.00           C  
ATOM     74  CD2 LEU A   9       6.600   7.600  19.500  1.00  0.00           C  
ATOM     75  N   VAL A  10       8.300  12.200  19.600  1.00  0.00           N  
ATOM     76  CA  VAL A  10       8.800  13.300  18.700  1.00  0.00           C  
ATOM     77  C   VAL A  10       9.800  14.200  19.500  1.00  0.00           C  
ATOM     78  O   VAL A  10      10.900  14.500  19.000  1.00  0.00           O  
ATOM     79  CB  VAL A  10       8.100  14.200  17.700  1.00  0.00           C  
ATOM     80  CG1 VAL A  10       8.900  15.400  17.300  1.00  0.00           C  
ATOM     81  CG2 VAL A  10       7.600  13.400  16.500  1.00  0.00           C      
    """.strip()
    handle1 = prody.parsePDBStream(io.StringIO(pdb1_string)).select("resnum > 1")
    full_pdb = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"))

    (aligned, rmsd) = proteopt.alignment.smart_align(
        handle1,
        full_pdb.select("resnum > 1 and resnum < 11"),
        reorder_atoms_by_name=False,
        fix_symmetries=False,
    )
    print(rmsd)
    assert rmsd > 0.5

    (aligned, rmsd) = proteopt.alignment.smart_align(
        handle1,
        full_pdb.select("resnum > 1 and resnum < 11"),
        reorder_atoms_by_name=True,
        fix_symmetries=False,
    )
    print(rmsd)
    assert rmsd < 1e-9


def test_atom_swaps():
    # The below extends the above example to also swap OE1 and OE2 from GLU 6
    pdb1_string = """
ATOM      1  N   VAL A   1      -2.900  17.600  15.500  1.00  0.00           N  
ATOM      2  CA  VAL A   1      -3.600  16.400  15.300  1.00  0.00           C  
ATOM      3  C   VAL A   1      -3.000  15.300  16.200  1.00  0.00           C  
ATOM      4  O   VAL A   1      -3.700  14.700  17.000  1.00  0.00           O  
ATOM      5  CB  VAL A   1      -3.500  16.000  13.800  1.00  0.00           C  
ATOM      6  CG1 VAL A   1      -2.100  15.700  13.300  1.00  0.00           C  
ATOM      7  CG2 VAL A   1      -4.600  14.900  13.400  1.00  0.00           C  
ATOM      8  N   LEU A   2      -1.700  15.100  16.000  1.00  0.00           N  
ATOM      9  C   LEU A   2      -1.000  13.900  18.300  1.00  0.00           C  
ATOM     10  O   LEU A   2      -0.900  14.900  19.000  1.00  0.00           O  
ATOM     11  CA  LEU A   2      -0.900  14.100  16.700  1.00  0.00           C  
ATOM     12  CB  LEU A   2       0.600  14.200  16.500  1.00  0.00           C  
ATOM     13  CG  LEU A   2       1.100  14.300  15.100  1.00  0.00           C  
ATOM     14  CD1 LEU A   2       0.400  15.500  14.400  1.00  0.00           C  
ATOM     15  CD2 LEU A   2       2.600  14.400  15.000  1.00  0.00           C  
ATOM     16  N   SER A   3      -1.100  12.600  18.600  1.00  0.00           N  
ATOM     17  CA  SER A   3      -1.100  12.200  20.000  1.00  0.00           C  
ATOM     18  C   SER A   3      -0.100  12.600  21.200  1.00  0.00           C  
ATOM     19  O   SER A   3       1.100  12.800  20.900  1.00  0.00           O  
ATOM     20  CB  SER A   3      -1.100  10.800  20.500  1.00  0.00           C  
ATOM     21  OG  SER A   3       0.200  10.100  20.300  1.00  0.00           O  
ATOM     22  N   GLU A   4      -0.700  12.600  22.400  1.00  0.00           N  
ATOM     23  CA  GLU A   4       0.000  12.900  23.600  1.00  0.00           C  
ATOM     24  C   GLU A   4       1.300  12.100  23.500  1.00  0.00           C  
ATOM     25  O   GLU A   4       2.400  12.600  23.600  1.00  0.00           O  
ATOM     26  CB  GLU A   4      -0.300  12.800  25.100  1.00  0.00           C  
ATOM     27  CG  GLU A   4       0.000  14.000  26.000  1.00  0.00           C  
ATOM     28  CD  GLU A   4       0.300  15.400  25.200  1.00  0.00           C  
ATOM     29  OE2 GLU A   4       1.200  16.000  25.400  1.00  0.00           O  
ATOM     30  OE1 GLU A   4      -0.600  15.500  24.300  1.00  0.00           O  
ATOM     31  N   GLY A   5       1.100  10.800  23.400  1.00  0.00           N  
ATOM     32  CA  GLY A   5       2.200   9.800  23.300  1.00  0.00           C  
ATOM     33  C   GLY A   5       3.200  10.200  22.200  1.00  0.00           C  
ATOM     34  O   GLY A   5       4.400  10.300  22.500  1.00  0.00           O  
ATOM     35  N   GLU A   6       2.700  10.400  21.000  1.00  0.00           N  
ATOM     36  CA  GLU A   6       3.500  10.700  19.800  1.00  0.00           C  
ATOM     37  C   GLU A   6       4.400  11.900  19.900  1.00  0.00           C  
ATOM     38  O   GLU A   6       5.500  11.900  19.400  1.00  0.00           O  
ATOM     39  CB  GLU A   6       2.600  10.700  18.600  1.00  0.00           C  
ATOM     40  CG  GLU A   6       2.000   9.400  18.100  1.00  0.00           C  
ATOM     41  CD  GLU A   6       0.900   9.500  17.000  1.00  0.00           C  
ATOM     42  OE1 GLU A   6       0.400   8.500  16.600  1.00  0.00           O  
ATOM     43  OE2 GLU A   6       0.700  10.600  16.700  1.00  0.00           O  
ATOM     44  N   TRP A   7       3.800  13.000  20.600  1.00  0.00           N  
ATOM     45  CA  TRP A   7       4.500  14.200  20.800  1.00  0.00           C  
ATOM     46  C   TRP A   7       5.700  13.700  21.700  1.00  0.00           C  
ATOM     47  O   TRP A   7       6.900  14.000  21.400  1.00  0.00           O  
ATOM     48  CB  TRP A   7       3.700  15.400  21.300  1.00  0.00           C  
ATOM     49  CG  TRP A   7       2.800  16.100  20.200  1.00  0.00           C  
ATOM     50  CD1 TRP A   7       1.500  16.200  20.100  1.00  0.00           C  
ATOM     51  CD2 TRP A   7       3.300  16.800  19.100  1.00  0.00           C  
ATOM     52  NE1 TRP A   7       1.100  16.900  18.900  1.00  0.00           N  
ATOM     53  CE2 TRP A   7       2.200  17.300  18.300  1.00  0.00           C  
ATOM     54  CE3 TRP A   7       4.600  17.100  18.600  1.00  0.00           C  
ATOM     55  CZ2 TRP A   7       2.300  18.100  17.200  1.00  0.00           C  
ATOM     56  CZ3 TRP A   7       4.700  17.900  17.500  1.00  0.00           C  
ATOM     57  CH2 TRP A   7       3.600  18.400  16.800  1.00  0.00           C  
ATOM     58  N   GLN A   8       5.400  12.900  22.700  1.00  0.00           N  
ATOM     59  CA  GLN A   8       6.300  12.300  23.600  1.00  0.00           C  
ATOM     60  C   GLN A   8       7.600  11.900  22.900  1.00  0.00           C  
ATOM     61  O   GLN A   8       8.700  12.300  23.200  1.00  0.00           O  
ATOM     62  CB  GLN A   8       6.300  12.200  25.100  1.00  0.00           C  
ATOM     63  CG  GLN A   8       7.600  12.200  25.700  1.00  0.00           C  
ATOM     64  CD  GLN A   8       7.700  12.200  27.200  1.00  0.00           C  
ATOM     65  OE1 GLN A   8       8.800  11.900  27.800  1.00  0.00           O  
ATOM     66  NE2 GLN A   8       6.600  12.500  27.800  1.00  0.00           N  
ATOM     67  N   LEU A   9       7.400  11.000  21.900  1.00  0.00           N  
ATOM     68  CA  LEU A   9       8.400  10.400  21.100  1.00  0.00           C  
ATOM     69  C   LEU A   9       9.100  11.400  20.300  1.00  0.00           C  
ATOM     70  O   LEU A   9      10.400  11.500  20.200  1.00  0.00           O  
ATOM     71  CB  LEU A   9       7.900   9.600  19.900  1.00  0.00           C  
ATOM     72  CG  LEU A   9       7.900   8.100  19.900  1.00  0.00           C  
ATOM     73  CD1 LEU A   9       8.200   7.700  21.400  1.00  0.00           C  
ATOM     74  CD2 LEU A   9       6.600   7.600  19.500  1.00  0.00           C  
ATOM     75  N   VAL A  10       8.300  12.200  19.600  1.00  0.00           N  
ATOM     76  CA  VAL A  10       8.800  13.300  18.700  1.00  0.00           C  
ATOM     77  C   VAL A  10       9.800  14.200  19.500  1.00  0.00           C  
ATOM     78  O   VAL A  10      10.900  14.500  19.000  1.00  0.00           O  
ATOM     79  CB  VAL A  10       8.100  14.200  17.700  1.00  0.00           C  
ATOM     80  CG1 VAL A  10       8.900  15.400  17.300  1.00  0.00           C  
ATOM     81  CG2 VAL A  10       7.600  13.400  16.500  1.00  0.00           C      
    """.strip()
    handle1 = prody.parsePDBStream(io.StringIO(pdb1_string)).select("resnum > 1")
    full_pdb = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"))

    (aligned, rmsd) = proteopt.alignment.smart_align(
        handle1,
        full_pdb.select("resnum > 1 and resnum < 11"),
        reorder_atoms_by_name=True,
        fix_symmetries=False,
    )
    print(rmsd)
    assert rmsd > 0.3

    (aligned, rmsd) = proteopt.alignment.smart_align(
        handle1,
        full_pdb.select("resnum > 1 and resnum < 11"),
        reorder_atoms_by_name=True,
        fix_symmetries=True,
    )
    print(rmsd)
    assert rmsd < 1e-9
