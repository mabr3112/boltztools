#!/usr/bin/env python3

"""Convert Boltz CCD pickle files to PDB files."""

import argparse
import pickle
from pathlib import Path

from rdkit import Chem


def find_pickle_files(input_path: Path) -> list[Path]:
    """Return the pickle files selected by an input file or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pkl":
            raise ValueError(f"Input file must have a .pkl extension: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.pkl"))
        if not files:
            raise ValueError(f"No .pkl files found in directory: {input_path}")
        return files
    raise ValueError(f"Input path does not exist: {input_path}")


def convert_pickle(input_file: Path, output_dir: Path) -> Path:
    """Write one pickled Boltz CCD molecule as a PDB file."""
    with input_file.open("rb") as handle:
        mol = pickle.load(handle)

    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"Pickle does not contain an RDKit molecule: {input_file}")

    # Copy Boltz's assigned atom names into the PDB atom-name field.
    for atom in mol.GetAtoms():
        if not atom.HasProp("name"):
            raise ValueError(f"Atom {atom.GetIdx()} has no Boltz 'name' property: {input_file}")
        atom_name = atom.GetProp("name")
        if len(atom_name) > 4:
            raise ValueError(f"Atom name exceeds the four-character PDB limit: {atom_name}")

        residue_info = atom.GetPDBResidueInfo() or Chem.AtomPDBResidueInfo()
        residue_info.SetName(atom_name.center(4))
        residue_info.SetResidueName(input_file.stem[:3].upper())
        residue_info.SetResidueNumber(1)
        residue_info.SetChainId("A")
        residue_info.SetIsHeteroAtom(True)
        atom.SetPDBResidueInfo(residue_info)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}.pdb"
    Chem.MolToPDBFile(mol, str(output_file))
    return output_file


def main() -> None:
    """Parse command-line arguments and convert the selected pickle files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Boltz .pkl file or directory of .pkl files")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory for output PDB files")
    args = parser.parse_args()

    try:
        input_files = find_pickle_files(args.input)
        for input_file in input_files:
            output_file = convert_pickle(input_file, args.output_dir)
            print(f"{input_file} -> {output_file}")
    except (OSError, pickle.UnpicklingError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
