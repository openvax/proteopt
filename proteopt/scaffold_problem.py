import collections
import logging
from collections import namedtuple
import numpy
import pickle

import prody

from .common import set_residue_data, serialize, deserialize
from .alignment import smart_align

class ConstrainedSegment(
        namedtuple(
            "ConstrainedSegment",
            ["length", "chain", "resindices", "sequence", "motif_num"])):
    def __eq__(self, other):
        print(self, other)
        return (
            self.length == other.length and
            self.chain == other.chain and
            numpy.array_equal(self.resindices, other.resindices) and
            self.sequence == other.sequence and
            self.motif_num == other.motif_num
        )

class UnconstrainedSegment(namedtuple("UnconstrainedSegment", ["length"])):
    pass

class ChainBreak(namedtuple("ChainBreak", [])):
    @property
    def length(self):
        return 0


def check_solution_sequence_is_valid(segments, sequence):
    # Recursive
    if not segments:
        if not sequence:
            return True, ""
        return False, "Sequence longer than segments"

    segment = segments[0]
    rest = segments[1:]
    if isinstance(segment, ConstrainedSegment):
        if segment.sequence is not None:
            sub_sequence = sequence[:segment.length]
            if segment.sequence.upper() != sub_sequence.upper():
                msg = (
                        "Invalid solution sequence. Sub-sequence: %s,"
                        "expected: %s." % (sub_sequence, segment.sequence))
                return False, msg
        return check_solution_sequence_is_valid(rest, sequence[segment.length:])
    if isinstance(segment, UnconstrainedSegment):
        (code, msg) = check_solution_sequence_is_valid(rest, sequence[segment.length:])
        if code:
            # Success
            return True, ""
        # No possible solution
        return False, "Not a valid solution."
    if isinstance(segment, ChainBreak):
        return check_solution_sequence_is_valid(rest, sequence)
    assert False


class ScaffoldProblem(object):
    def __init__(self, structure, segments=None):
        self.structure = structure

        if segments is None:
            segments = []
        self.segments = segments

    def get_first_chain(self):
        segments = []
        for segment in self.segments:
            if isinstance(segment, ChainBreak):
                break
            segments.append(segment)
        return ScaffoldProblem(self.structure, segments)

    def get_structure(self, renumber_resnums_as_resindices=True):
        handle = self.structure.copy()
        if renumber_resnums_as_resindices:
            handle.setResnums(handle.getResindices() + 1)
        return handle

    def check_solution_sequence_is_valid(self, sequence, ipdb_on_error=False):
        code, msg = check_solution_sequence_is_valid(self.segments, sequence)
        if code:
            return True
        if ipdb_on_error:
            import ipdb
            ipdb.set_trace()
        raise ValueError("%s Full sequence: %s. Segments: %s" % (
            msg, sequence, self.segments))

    def annotate_solution(self, structure):
        masks = collections.defaultdict(list)
        data_items = collections.defaultdict(list)
        for (segment_num, segment) in enumerate(self.segments):
            if isinstance(segment, (UnconstrainedSegment, ConstrainedSegment)):
                mask_item = {
                    "constrained_by_structure": (
                        isinstance(segment, ConstrainedSegment) and
                        segment.resindices is not None),
                    "constrained_by_sequence": (
                        isinstance(segment, ConstrainedSegment) and
                        segment.sequence is not None),
                    "unconstrained": isinstance(segment, UnconstrainedSegment),
                }
                for (k, v) in mask_item.items():
                    masks[k].extend([v] * segment.length)

                data_items["scaffold_problem_segment"].extend(
                    [segment_num] * segment.length)
            else:
                continue
        for (k, v) in masks.items():
            set_residue_data(structure, k, v, is_bool=True)

        for (k, v) in data_items.items():
            set_residue_data(structure, k, v, is_bool=False)

        data_items.update(masks)
        return data_items

    def add_segment(
            self,
            length=None,     # specify only length for an unconstrained segment
            structure=None,  # constrain by coordinates
            sequence=None,   # constrain by sequence if specified
            sequence_from_structure=True,  # if structure specified use sequence from that
            motif_num=None,  # for grouping motifs
    ):
        # Add a segment with a fixed structure, sequence, or both.
        # sub_structure must be taken from self.structure (i.e. resindices must
        # agree)
        resindices = None
        chain = None
        if structure is not None:
            chains = numpy.unique(structure.getChids())
            if len(chains) > 1:
                raise NotImplementedError("Multiple chains in contig")
            chain, = chains
            resindices = structure.ca.getResindices()
            if length is None:
                length = len(resindices)
            assert len(resindices) == length
            if sequence is None and sequence_from_structure:
                sequence = structure.ca.getSequence()
        if sequence is not None:
            if length is not None and length != len(sequence):
                raise ValueError(
                    "structure has %d residues but sequence has %d" % (
                        len(resindices), len(sequence)))
            if length is None:
                length = len(sequence)
            assert len(sequence) == length
        if length is None:
            raise ValueError("must specify length, structure, or sequence")

        segment = None
        if resindices is None and sequence is None:
            segment = UnconstrainedSegment(length=length)
        else:
            segment = ConstrainedSegment(
                length, chain, resindices, sequence, motif_num=motif_num)

        self.segments.append(segment)
        return self

    def add_contig_chain_break(self):
        self.segments.append(ChainBreak())
        return self

    def constrained_segments(self):
        return [s for s in self.segments if isinstance(s, ConstrainedSegment)]

    def motif_nums(self):
        motif_nums = set()
        for segment in self.constrained_segments():
            if segment.motif_num is None:
                raise ValueError("No motif_num specified for %s" % str(segment))
            motif_nums.add(segment.motif_num)
        motif_nums = sorted(motif_nums)
        return motif_nums

    def evaluate_solution(self, designed_structure, prefix=""):
        designed_structure = designed_structure.copy()  # resets resindices
        constrained_segments = self.constrained_segments()
        motif_nums = self.motif_nums()
        results = collections.defaultdict(list)
        for motif_num in motif_nums:
            # There must be a more efficient way to do this
            # Note: this is complicated because order matters

            # Collect target resindices
            target_resindices = []
            for segment in constrained_segments:
                if segment.motif_num == motif_num:
                    target_resindices.extend(segment.resindices)

            # Run-length encode target resindices
            target_resindices_start_stop = []
            for idx in target_resindices:
                if (
                        target_resindices_start_stop and
                        idx == target_resindices_start_stop[-1][1] + 1):
                    target_resindices_start_stop[-1][1] = idx
                else:
                    target_resindices_start_stop.append([idx, idx])

            reference_target_pieces = []
            for (start, stop) in target_resindices_start_stop:
                reference_target_piece = self.get_structure().select(
                    "resindex %d to %d" % (start, stop)).copy()
                reference_target_pieces.append(reference_target_piece)
            reference_target = combine_atom_groups(reference_target_pieces)

            designed_target_pieces = []
            designed_resindices = designed_structure.getResindices()
            numpy.testing.assert_equal(
                designed_structure.ca.getResindices(),
                numpy.arange(max(designed_resindices) + 1)
            )
            current_index = 0
            for segment in self.segments:
                if isinstance(segment, ConstrainedSegment) and segment.motif_num == motif_num:
                    new_piece = designed_structure.select(
                        f"resindex {current_index} to {current_index + segment.length - 1}"
                    ).copy()
                    designed_target_pieces.append(new_piece)
                current_index += segment.length
            designed_target = combine_atom_groups(designed_target_pieces)

            # We allow missing atoms in reference_target
            missing_in_reference = [
                pair for pair
                in zip(designed_target.getResindices(), designed_target.getNames())
                if pair not in set(
                    zip(reference_target.getResindices(), reference_target.getNames()))
            ]
            if missing_in_reference:
                logging.warning("Omitting missing atoms from reference: %s", str(missing_in_reference))
                sel = "not (%s)" % " or ".join(
                    f"(resindex {i} and name {name})" for (i, name) in missing_in_reference)
                designed_target = designed_target.select(sel).copy()

            numpy.testing.assert_equal(len(reference_target.ca), len(designed_target.ca))
            numpy.testing.assert_equal(len(reference_target), len(designed_target))
            ca_rmsd = smart_align(reference_target.ca, designed_target.ca).rmsd
            all_atom_rmsd = smart_align(reference_target, designed_target).rmsd
            results[f"motif_{motif_num}_ca_rmsd"] = ca_rmsd
            results[f"motif_{motif_num}_all_atom_rmsd"] = all_atom_rmsd

        renamed_results = {}
        for (key, value) in results.items():
            renamed_results[prefix + key] = value

        return renamed_results

    def __repr__(self):
        return(
            "<ScaffoldProblem segments=\n%s\n>" % "\n".join(str(s) for s in self.segments))

    def __eq__(self, other):
        return (
            self.structure == other.structure and
            self.segments == other.segments)

    def __str__(self):
        return repr(self)


def combine_atom_groups(pieces):
    result = pieces[0].copy()
    for piece in pieces[1:]:
        result += piece
        result._title = "Combined"
    return result.copy()
