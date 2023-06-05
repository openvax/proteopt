import tempfile
import os
import json
from typing import Optional

import numpy
import pandas

import prody

import yabul

from .common import args_from_function_signature

class ProteinMPNN(object):
    tool_name = "proteinmpnn"
    config_args = {}
    model_args = {}

    def __init__(self):
        pass

    def run(
            self,
            structure : prody.Atomic,
            fixed : Optional[prody.Atomic] = None,
            num : int = 1,
            ca_only : bool = False,
            sampling_temp : float = 0.1,
            batch_size : int = 1,
            verbose : bool = False):

        from ProteinMPNN import protein_mpnn_run

        # Reset resnums to avoid gaps
        chains = numpy.unique(structure.getChids())
        structure = structure.copy()
        original_resnums = {}
        for chain in chains:
            selected = structure.select("chain %s" % chain)
            original_resnums[chain] = selected.ca.getResnums()
            mapped = dict(
                (k, i)
                for (i, k) in
                enumerate(numpy.unique(selected.getResindices())))
            selected.setResnums(
                [mapped[original] for original in selected.getResindices()]
            )

        temp_dir = tempfile.TemporaryDirectory("proteopt_proteinmpnn")
        input_pdb_path = os.path.join(temp_dir.name, "input.pdb")
        prody.writePDB(input_pdb_path, structure)

        out_path = os.path.join(temp_dir.name, "results")
        os.mkdir(out_path)

        fixed_positions_jsonl = None
        if fixed:
            chain_to_fixed_indices_1p = {}
            for chain in chains:
                fixed_chain = fixed.select("chain %s" % chain)
                if fixed_chain is None:
                    # No such chain in fixed
                    chain_to_fixed_indices_1p[chain] = []
                else:
                    fixed_resnums = fixed_chain.ca.getResnums()
                    resnum_to_index = dict(
                        (resnum, i)
                        for (i, resnum)
                        in enumerate(original_resnums[chain]))
                    chain_to_fixed_indices_1p[chain] = [
                        resnum_to_index[resnum] + 1
                        for resnum in fixed_resnums
                    ]

            fixed_json_dict = {
                "input": chain_to_fixed_indices_1p
            }
            fixed_positions_jsonl = os.path.join(
                temp_dir.name, "fixed_positions.json")
            with open(fixed_positions_jsonl, "w") as fd:
                json.dump(fixed_json_dict, fd)
                fd.write("\n")

        argv = [
            "--pdb_path", input_pdb_path,
            "--num_seq_per_target", num,
            "--save_probs", "1",
            "--batch_size", batch_size,
            "--sampling_temp", sampling_temp,
            "--out_folder", out_path,
            "--suppress_print", ("0" if verbose else "1"),
            # Todo: support:
            # seed, omit_AAs, pssm_jsonl, tied_positions_jsonl, pssm_multi
        ]
        if ca_only:
            argv.append("--ca_only")

        if fixed_positions_jsonl:
            argv.append("--fixed_positions_jsonl")
            argv.append(fixed_positions_jsonl)

        argv = [str(x) for x in argv]

        parsed_args = protein_mpnn_run.argparser.parse_args(argv)
        protein_mpnn_run.main(parsed_args)

        results = yabul.read_fasta(
            os.path.join(out_path, "seqs", "input.fa"))

        result_df = pandas.DataFrame(
            results.description.str.split(",").map(
                lambda pieces: [piece.split("=") for piece in pieces]
            ).map(dict).tolist())
        result_df["seq"] = results.sequence.values

        seq_split = result_df.seq.str.split("/")
        for (i, chain) in enumerate(chains):
            result_df["seq_%s" % chain] = seq_split.str.get(i)

        probs = dict(
            numpy.load(os.path.join(out_path, "probs", "input.npz")))

        result_df["probs"] = list(probs["probs"])
        result_df["log_probs"] = list(probs["log_probs"])

        result_df.columns = result_df.columns.str.strip()

        temp_dir.cleanup()
        return result_df

    def run_multiple(self, list_of_dicts, show_progress=False):
        results = []

        iterator = list_of_dicts
        if show_progress:
            import tqdm
            iterator = tqdm.tqdm(list_of_dicts)

        for kwargs in iterator:
            result = self.run(**kwargs)
            assert result is not None
            results.append(result)
        return results

    run_args = args_from_function_signature(run)

