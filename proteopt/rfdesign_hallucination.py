import tempfile
import os
import pickle

import numpy
import pandas

import prody

from .common import set_residue_data
from rfdesign.hallucination import hallucinate

from .scaffold_problem import (
    ScaffoldProblem, UnconstrainedSegment, ConstrainedSegment, ChainBreak)


def make_args_from_problem(problem: ScaffoldProblem):
    mask = []
    force_aa_hal = []

    hallucinated_index = 1
    for (i, contig) in enumerate(problem.segments):
        if isinstance(contig, ConstrainedSegment):
            to_append = None
            if contig.sequence is not None:
                for (i, aa) in enumerate(contig.sequence):
                    # We always use chain A here since it is the designed chain
                    force_aa_hal.append(
                        "A%d%s" % (hallucinated_index + i, aa))

            if contig.resindices is not None:
                previous = None
                pieces = []
                for resnum in contig.resindices + 1:
                    if previous is None:
                        # Start segment
                        pieces.append("%s%d-" % (contig.chain, resnum))
                    elif resnum != previous + 1:
                        # End segment
                        pieces.append("%d" % previous)
                        pieces.append(",%s%d-" % (contig.chain, resnum))
                    previous = resnum
                # End final segment
                pieces.append("%d" % previous)
                mask.append("".join(pieces))
            else:
                mask.append("%d" % contig.length)
            hallucinated_index += contig.length
        elif isinstance(contig, UnconstrainedSegment):
            raise NotImplementedError("VariableLengthSegment")
        elif isinstance(contig, ChainBreak):
            raise NotImplementedError("chain break")
        else:
            raise ValueError(contig)

    return [
        "--mask", ",".join(mask),
        "--force_aa_hal", ",".join(force_aa_hal),
    ]


class RFDesignHallucination(object):
    def __init__(self, device='cuda:0', num_threads=4):
        self.device = device
        self.num_threads = num_threads
        self.initialization = hallucinate.initialize(
            device=device, n_threads=num_threads)

    def run(
            self,
            problem : ScaffoldProblem,
            num,
            steps,
            w_rog=0,
            w_crmsd=0,
            learning_rate=0.05,
            drop=0.2,
            extra_args=[],
            add_extra_losses_function=None):

        temp_dir = tempfile.TemporaryDirectory("proteopt_rfdesign_hallucination")
        input_pdb_path = os.path.join(temp_dir.name, "input.pdb")
        prody.writePDB(input_pdb_path, problem.get_structure())

        out_path = os.path.join(temp_dir.name, "results")
        os.mkdir(out_path)

        argv = [
            "--pdb", input_pdb_path,
            "--num", num,
            "--steps", steps,
            "--w_rog", w_rog,
            "--w_crmsd", w_crmsd,
            "--learning_rate", learning_rate,
            "--drop", drop,
            "--device", self.device,
            "--nthreads", self.num_threads,
            "--out", os.path.join(out_path, "result"),
        ] + extra_args
        argv.extend(make_args_from_problem(problem))
        argv = [str(x) for x in argv]

        hallucinate.main(
            argv=argv,
            initialization_result=self.initialization,
            add_extra_losses_function=add_extra_losses_function)

        df_result_filenames = pandas.DataFrame({
            "filename": os.listdir(out_path),
        })
        splitted = df_result_filenames.filename.str.split(".", regex=False)
        df_result_filenames["run"] = splitted.str.get(0)
        df_result_filenames["extension"] = splitted.str.get(1)

        df_result_pivoted = df_result_filenames.pivot(
            index="run", columns="extension", values="filename").applymap(
            lambda s: os.path.join(out_path, s))

        df_result = pandas.DataFrame(index=df_result_pivoted.index)
        df_result["structure"] = df_result_pivoted.pdb.map(prody.parsePDB)

        def load_pickle(filename):
            with open(filename, "rb") as fd:
                return pickle.load(fd)

        df_result["metadata"] = df_result_pivoted.trb.map(load_pickle)
        df_result["settings"] = df_result_pivoted.trk.map(load_pickle)
        # df_result["values"] = df_result_pivoted.npz.map(
        #    lambda s: dict(numpy.load(s)))

        # Pull out some values from metadata
        for item in df_result.iloc[0].metadata:
            if item.startswith("loss_"):
                values = []
                for d in df_result.metadata:
                    (value,) = d[item]
                    values.append(value)
                df_result[item] = values

        # Update structures with extra info
        for _, row in df_result.iterrows():
            # Original chain and resnums
            chain_and_resnum_new_to_original = dict(
                zip(row.metadata['con_hal_pdb_idx'], row.metadata['con_ref_pdb_idx']))
            original_chain_and_resnums = [
                chain_and_resnum_new_to_original.get(new, ("X", -1))
                for new in zip(row.structure.getChids(), row.structure.getResnums())
            ]
            row.structure.setData(
                "original_chain",
                [chain for (chain, resnum) in original_chain_and_resnums])
            row.structure.setData(
                "original_resnum",
                [resnum for (chain, resnum) in original_chain_and_resnums])

            problem.annotate_solution(row.structure)

        for structure in df_result.structure:
            seq = structure.ca.getSequence()
            problem.check_solution_sequence_is_valid(seq)  # throws

        temp_dir.cleanup()
        return df_result.reset_index(drop=True)



