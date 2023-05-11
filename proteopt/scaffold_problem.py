import collections
import numpy

from .common import set_residue_data

ConstrainedSegment = collections.namedtuple(
    "ConstrainedSegment", ["length", "chain", "resnums", "sequence"])

GeneratedSegment = collections.namedtuple(
    "GeneratedSegment", ["min_length", "max_length"])

ChainBreak = collections.namedtuple("ChainBreak", [])


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
    if isinstance(segment, GeneratedSegment):
        for length in range(segment.min_length, segment.max_length + 1):
            (code, msg) = check_solution_sequence_is_valid(rest, sequence[length:])
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

    def is_fixed_length(self):
        return all([
            not isinstance(s, GeneratedSegment)
            for s in self.segments
        ])

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
        if not self.is_fixed_length():
            raise ValueError("No defined length")

        masks = collections.defaultdict(list)
        data_items = collections.defaultdict(list)
        for (segment_num, segment) in enumerate(self.segments):
            if isinstance(segment, ConstrainedSegment):
                mask_item = {
                    "constrained_by_structure": segment.resnums is not None,
                    "constrained_by_sequence": segment.sequence is not None,
                    "unconstrained": (
                            segment.resnums is None
                            and segment.sequence is None)
                }
                for (k, v) in mask_item.items():
                    masks[k].extend([v] * segment.length)

                data_items["scaffold_problem_segment"].extend(
                    [segment_num] * segment.length)

            elif isinstance(segment, GeneratedSegment):
                assert False
            else:
                continue
        for (k, v) in masks.items():
            set_residue_data(structure, k, v, is_bool=True)

        for (k, v) in data_items.items():
            set_residue_data(structure, k, v, is_bool=False)

        data_items.update(masks)
        return data_items

    def add_fixed_length_segment(
            self,
            length=None,
            structure=None,  # constrain by coordinates
            sequence=None,  # constrain by sequence if specified
            sequence_from_structure=True,  # if structure specified use sequence from that
    ):
        # Add a segment with a fixed structure, sequence, or both.
        # sub_structure must be taken from self.structure (i.e. resnums must
        # agree)
        resnums = None
        chain = None
        if structure is not None:
            chains = numpy.unique(structure.getChids())
            if len(chains) > 1:
                raise NotImplementedError("Multiple chains in contig")
            chain, = chains
            resnums = structure.ca.getResnums()
            if length is None:
                length = len(resnums)
            assert len(resnums) == length
            if sequence is None and sequence_from_structure:
                sequence = structure.ca.getSequence()
        if sequence is not None:
            if length is not None and length != len(sequence):
                raise ValueError(
                    "structure has %d residues but sequence has %d" % (
                        len(resnums), len(sequence)))
            if length is None:
                length = len(sequence)
            assert len(sequence) == length
        if length is None:
            raise ValueError("must specify length, structure, or sequence")
        self.segments.append(ConstrainedSegment(length, chain, resnums, sequence))
        return self

    def add_variable_length_segment(self, min_length, max_length):
        if min_length == max_length:
            return self.add_fixed_length_segment(length=min_length)
        self.segments.append(GeneratedSegment(min_length, max_length))
        return self

    def add_contig_chain_break(self):
        self.segments.append(ChainBreak())
        return self

    def fixed_length_problems(self, num_to_sample=None):
        variable_segments_to_sample = {}
        total_possible = 1
        for (i, segment) in enumerate(self.segments):
            if isinstance(segment, GeneratedSegment):
                variable_segments_to_sample[i] = segment
                total_possible *= segment.max_length - segment.min_length + 1

        if num_to_sample is None:
            num_to_sample = total_possible
        if num_to_sample > total_possible:
            raise ValueError("Can't sample %d, total possible is %d" % (
                num_to_sample, total_possible))

        if variable_segments_to_sample:
            already_sampled = set()
            for loops in range(1000000):
                # Pick a segment to sample
                i = numpy.random.choice(list(variable_segments_to_sample))
                segment = variable_segments_to_sample[i]
                sampled_length = numpy.random.randint(
                    segment.min_length, segment.max_length + 1)
                new_object = ScaffoldProblem(self.structure)
                new_object.segments = (
                        self.segments[:i]
                        + [
                            ConstrainedSegment(sampled_length, None, None, None)
                        ]
                        + self.segments[i + 1:])
                fully_fixed = next(new_object.fixed_length_problems())
                all_sampled_lengths = tuple(
                    fully_fixed.segments[i]
                    for i in variable_segments_to_sample.keys())

                if all_sampled_lengths not in already_sampled:
                    already_sampled.add(all_sampled_lengths)
                    yield fully_fixed
                if num_to_sample and len(already_sampled) == num_to_sample:
                    break
            else:
                raise ValueError("Too many iterations")
        else:
            # We are already a fixed-length problem
            yield self

    def __repr__(self):
        return(
            "<ScaffoldProblem segments=\n%s\n>" % "\n".join(str(s) for s in self.segments))

    def __str__(self):
        return repr(self)