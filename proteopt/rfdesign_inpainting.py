import tempfile
import os
import pickle

import numpy
import pandas

import prody

from rfdesign.inpainting import inpaint

from .common import set_residue_data
from .scaffold_problem import (
    ScaffoldProblem, GeneratedSegment, ConstrainedSegment, ChainBreak)


def make_contigs_argument(problem : ScaffoldProblem):
    contigs = []
    for (i, segment) in enumerate(problem.segments):
        next_contig = (
            None if i == len(problem.segments) - 1 else problem.segments[i + 1])
        if isinstance(segment, ConstrainedSegment):
            if segment.resnums is not None:
                previous = None
                pieces = []
                for resnum in segment.resnums:
                    if previous is None:
                        # Start segment
                        pieces.append("%s%d-" % (segment.chain, resnum))
                    elif resnum != previous + 1:
                        # End segment
                        pieces.append("%d" % previous)
                        pieces.append(",%s%d-" % (segment.chain, resnum))
                    previous = resnum

                # End final segment
                pieces.append("%d" % previous)
                contigs.append("".join(pieces))
                contigs.append(
                    ",0 " if isinstance(next_contig, ChainBreak) else ",")

                # Implementation limitation:
                # Structure sequence has to match specified sequence
                structure_sequence = problem.structure.select(
                    "chain %s and resnum %s" % (
                        segment.chain, " ".join(str(x) for x in segment.resnums))
                ).ca.getSequence()
                if segment.sequence != structure_sequence:
                    raise NotImplementedError(
                        "Structure sequence must match constrained sequence",
                        [segment.sequence, structure_sequence])

            else:
                contigs.append("%d," % segment.length)
        elif isinstance(segment, GeneratedSegment):
            contigs.append("%d-%d" % (segment.min_length, segment.max_length))
            contigs.append(
                ",0 " if isinstance(next_contig, ChainBreak) else ",")
        elif isinstance(segment, ChainBreak):
            pass
        else:
            raise ValueError(segment)

    if contigs[-1] in (",", ",0 "):
        contigs = contigs[:-1]

    return "".join(contigs).strip(',')


class RFDesignInpainting(object):
    def __init__(self, device=None, num_threads=4):
        self.device = device
        self.num_threads = num_threads
        self.initialization = inpaint.initialize(device=device)

    def run(
            self,
            problem : ScaffoldProblem,
            num=1,
            n_cycle=4):

        temp_dir = tempfile.TemporaryDirectory("proteopt_rfdesign_inpainting")
        input_pdb_path = os.path.join(temp_dir.name, "input.pdb")
        prody.writePDB(input_pdb_path, problem.structure)

        out_path = os.path.join(temp_dir.name, "results")
        os.mkdir(out_path)

        argv = [
            "--pdb", input_pdb_path,
            "--num_designs", num,
            "--dump_pdb",
            "--dump_trb",
            "--n_cycle", n_cycle,
            "--contigs", make_contigs_argument(problem),
            "--out", os.path.join(out_path, "result"),
        ]
        argv = [str(x) for x in argv]
        print(argv)

        inpaint.main(
            argv=argv,
            initialization_result=self.initialization)

        df_result_filenames = pandas.DataFrame({
            "filename": os.listdir(out_path),
        })
        df_result_filenames = df_result_filenames.loc[
            df_result_filenames.filename != "FLAGS.txt"
        ].copy()
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

            # lDDT
            set_residue_data(
                row.structure, "inpaint_lddt", row.metadata['lddt'])

            # Masked
            set_residue_data(
                row.structure,
                "inpaint_masked",
                row.metadata['mask_1d'],
                is_bool=True)

            # Extra masks from problem
            if problem.is_fixed_length():
                problem.annotate_solution(row.structure)

        # Add sequences to dataframe
        def get_sequence(single_chain_ca):
            amino_acids = numpy.array(list(single_chain_ca.getSequence()))
            masked = single_chain_ca.getFlags("inpaint_masked")
            amino_acids[~masked] = [s.lower() for s in amino_acids[~masked]]
            return "".join(amino_acids)

        chains = numpy.unique(df_result.structure.iloc[0].getChids())
        for chain in chains:
            df_result["seq_%s" % chain] = df_result.structure.map(
                lambda structure: get_sequence(
                    structure.select("ca and chain %s" % chain)))

        if len(chains) == 1:
            chain, = chains
            for seq in df_result["seq_%s" % chain]:
                problem.check_solution_sequence_is_valid(seq)  # throws

        temp_dir.cleanup()
        return df_result.reset_index(drop=True)



