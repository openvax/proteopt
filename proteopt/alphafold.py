# Note: much of this is inspired by RFDesign af2_metrics.py script
import os
# See https://github.com/google/jax/issues/1222
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.5'

import io
import numpy
import pandas
import prody
import yabul

from typing import Optional

from .common import args_from_function_signature

prody.confProDy(verbosity='none')

# Adapted from RFDesign (BSD license)
# https://github.com/RosettaCommons/RFDesign
def make_fixed_size(inputs, shape_schema, msa_cluster_size, extra_msa_size,
                    num_res, num_templates=0):
    """Guess at the MSA and sequence dimensions to make fixed size."""
    from alphafold.model.tf import shape_placeholders
    import tensorflow.compat.v1 as tf

    NUM_RES = shape_placeholders.NUM_RES
    NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
    NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
    NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES

    pad_size_map = {
      NUM_RES: num_res,
      NUM_MSA_SEQ: msa_cluster_size,
      NUM_EXTRA_SEQ: extra_msa_size,
      NUM_TEMPLATES: num_templates,
    }

    for k, v in inputs.items():
      if k == 'extra_cluster_assignment':
          continue
      shape = list(v.shape)
      schema = shape_schema[k]
      assert len(shape) == len(schema), (
          f'Rank mismatch between shape and shape schema for {k}: '
          f'{shape} vs {schema}')
      pad_size = [
          pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)
      ]
      padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]
      if padding:
          inputs[k] = tf.pad(
              v, padding, name=f'pad_to_fixed_{k}')
          inputs[k].set_shape(pad_size)
    return {k: numpy.asarray(v) for k, v in inputs.items()}


class AlphaFold(object):
    tool_name = "alphafold"

    def __init__(
            self,
            data_dir : str,
            max_length : int = 512,
            model_name : str = "model_4_ptm",
            num_recycle : int = 3,
            amber_relax : bool = True):

        self.model_name = model_name
        if model_name == "MOCK":
            return

        from alphafold.model import data, config, model, utils

        self.max_length = max_length
        self.amber_relax = amber_relax

        self.model_config = config.model_config(model_name)
        self.model_config.data.eval.num_ensemble = 1
        self.model_config.model.num_recycle = num_recycle
        self.model_config.data.common.num_recycle = num_recycle

        # Not setting these:
        # self.model_config.data.common.max_extra_msa = 1
        # self.model_config.data.eval.max_msa_clusters = 1

        weights_path = os.path.join(data_dir, f'params_{model_name}.npz')
        with open(weights_path, 'rb') as f:
            params = numpy.load(io.BytesIO(f.read()), allow_pickle=False)
        self.model_params = utils.flat_params_to_haiku(params)
        self.model_runner = model.RunModel(self.model_config, self.model_params)
        self.model2crop_feats = {
            k:[None]+v for k,v in dict(self.model_config.data.eval.feat).items()
        }

    config_args = args_from_function_signature(__init__, include=["data_dir"])
    model_args = args_from_function_signature(__init__, exclude=list(config_args))

    def run_multiple(self, list_of_dicts, show_progress=False):
        results = []

        iterator = list_of_dicts
        if show_progress:
            import tqdm
            iterator = tqdm.tqdm(list_of_dicts)

        for kwargs in iterator:
            if isinstance(kwargs, str):
                kwargs = {"sequence": kwargs}
            result = self.run(**kwargs)
            assert result is not None
            results.append(result)
        return results

    def run(
            self,
            sequence : str,
            template : Optional[prody.Atomic] = None,
            template_replace_sequence_with_gaps : bool = True,
            template_mask_sidechains : bool = True,
            b_factor_source : str = "experimentally_resolved"):

        if self.model_name == "MOCK":
            return MockAlphaFoldModel.predict(sequence)

        from alphafold.common import protein, residue_constants
        from alphafold.data import pipeline, templates, parsers
        from alphafold.relax import relax
        import tensorflow.compat.v1 as tf

        L = len(sequence)

        # Handle template
        if template is not None:
            # Some borrowed from https://github.com/jproney/AF2Rank/blob/master/test_templates.py
            stream = io.StringIO()
            prody.writePDBStream(stream, template.select("protein"))
            stream.seek(0)
            pdb_string = stream.read()
            template_protein = protein.from_pdb_string(pdb_string)

            template_sequence = template.ca.getSequence()

            numpy.testing.assert_equal(
                template_protein.atom_mask.shape,
                (len(template_sequence), 37))

            template_atom_mask = template_protein.atom_mask
            template_positions = template_protein.atom_positions

            if template_mask_sidechains:
                # This leaks information on glycines
                template_atom_mask[5:] = 0
                template_positions[5:] = 0

            # Alignment
            alignment = yabul.align_pair(
                query_seq=sequence,
                reference_seq=template_sequence,
                local=False)

            template_to_sequence_mapping = {}
            template_pos = 0
            seq_pos = 0
            template_aligned_sequence = []
            for template_char, seq_char in zip(alignment.reference, alignment.query):
                if seq_char == '-':
                    template_pos += 1
                elif template_char == '-':
                    seq_pos += 1
                    template_aligned_sequence.append(template_char)
                else:
                    template_to_sequence_mapping[template_pos] = seq_pos
                    seq_pos += 1
                    template_pos += 1
                    template_aligned_sequence.append(template_char)

            template_aligned_sequence = "".join(template_aligned_sequence)
            template_to_sequence_mapping = pandas.Series(template_to_sequence_mapping)

            new_template_positions = numpy.zeros((L, 37, 3))
            new_template_atom_mask = numpy.zeros((L, 37))

            new_template_positions[template_to_sequence_mapping.values] = template_positions[
                template_to_sequence_mapping.index.values
            ]
            new_template_atom_mask[template_to_sequence_mapping.values] = template_atom_mask[
                template_to_sequence_mapping.index.values
            ]

            effective_template_squence = (
                    "-" * len(template_aligned_sequence)
                    if template_replace_sequence_with_gaps
                    else template_aligned_sequence
                )

            template_features = {
                "template_aatype": residue_constants.sequence_to_onehot(
                    effective_template_squence,
                    residue_constants.HHBLITS_AA_TO_ID)[None],
                "template_sequence": template_aligned_sequence.replace("-", ""),
                "template_all_atom_masks": new_template_atom_mask[None],
                "template_all_atom_positions": new_template_positions[None],
                "template_domain_names": numpy.asarray(["None"]),
                # Omitting "template_sum_probs"
            }
        else:
            template_features = {}

        msa_alignment = parsers.parse_a3m(">query\n%s" % sequence)

        # Prepare input
        feature_dict = {
            **pipeline.make_sequence_features(
                sequence=sequence, description="none", num_res=L),
            **pipeline.make_msa_features(
                [msa_alignment]),
            **template_features,
        }

        inputs = self.model_runner.process_features(feature_dict, random_seed=0)

        inputs_padded = make_fixed_size(
            inputs,
            self.model2crop_feats,
            msa_cluster_size=0,
            extra_msa_size=0,
            num_res=self.max_length,
            num_templates=4 if template is not None else 0)

        # Run AF2
        outputs = self.model_runner.predict(inputs_padded, random_seed=42)

        # Process results
        if b_factor_source == "experimentally_resolved":
            b_factors = 1 - tf.sigmoid(
                outputs["experimentally_resolved"]["logits"]).numpy()[:L]
        elif b_factor_source is None:
            b_factors = numpy.zeros_like(
                outputs['structure_module']['final_atom_mask'])[:L]
        else:
            raise ValueError("Unsupported b_factor_source: %s" % b_factor_source)

        unrelaxed_protein = protein.Protein(
            aatype=inputs_padded['aatype'][0][:L],
            atom_positions=outputs['structure_module']['final_atom_positions'][:L],
            atom_mask=outputs['structure_module']['final_atom_mask'][:L],
            residue_index=inputs_padded['residue_index'][0][:L] + 1,
            b_factors=b_factors,
            chain_index=numpy.zeros(L),
        )

        if self.amber_relax:
            amber_relaxer = relax.AmberRelaxation(
                max_iterations=0,
                tolerance=2.39,
                stiffness=10.0,
                exclude_residues=[],
                max_outer_iterations=20,
                use_gpu=True,
            )
            pdb_lines, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
            title = "AF2 prediction (relaxed, %s)" % self.model_name
        else:
            pdb_lines = protein.to_pdb(unrelaxed_protein)
            title = "AF2 prediction (unrelaxed, %s)" % self.model_name

        handle = prody.parsePDBStream(
            io.StringIO(pdb_lines),
            title=title)

        plddt = pandas.Series(outputs['plddt'][:len(sequence)])
        assert set(handle.getResindices()) == set(plddt.index)
        res_indices = pandas.Series(handle.getResindices())
        handle.setData(
            "af2_plddt", res_indices.map(plddt))

        if "ptm" in outputs:
            handle.setData("af2_ptm", float(outputs["ptm"]))

        return handle

    run_args = args_from_function_signature(run)

    # For backwards compatability
    predict = run


class MockAlphaFoldModel():
    """Mock AlphaFold model for testing"""

    MOCK_PDB = """
    CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1
    ATOM      1  N   ALA B  27     105.673 159.516 139.644  1.00 60.56      B    N  
    ATOM      2  CA  ALA B  27     105.810 159.161 141.050  1.00 60.56      B    C  
    ATOM      3  C   ALA B  27     107.274 158.949 141.411  1.00 60.56      B    C  
    ATOM      4  O   ALA B  27     108.140 158.939 140.540  1.00 60.56      B    O  
    ATOM      5  CB  ALA B  27     105.000 157.916 141.366  1.00 60.56      B    C  
    ATOM      6  N   TYR B  28     107.544 158.787 142.704  1.00 54.45      B    N  
    ATOM      7  CA  TYR B  28     108.895 158.571 143.197  1.00 54.45      B    C  
    ATOM      8  C   TYR B  28     108.856 157.549 144.321  1.00 54.45      B    C  
    ATOM      9  O   TYR B  28     107.834 157.371 144.987  1.00 54.45      B    O  
    ATOM     10  CB  TYR B  28     109.532 159.869 143.707  1.00 54.45      B    C  
    ATOM     11  CG  TYR B  28     109.692 160.944 142.659  1.00 54.45      B    C  
    ATOM     12  CD1 TYR B  28     110.823 160.993 141.855  1.00 54.45      B    C  
    ATOM     13  CD2 TYR B  28     108.716 161.915 142.481  1.00 54.45      B    C  
    ATOM     14  CE1 TYR B  28     110.975 161.979 140.899  1.00 54.45      B    C  
    ATOM     15  CE2 TYR B  28     108.858 162.904 141.527  1.00 54.45      B    C  
    ATOM     16  CZ  TYR B  28     109.989 162.930 140.740  1.00 54.45      B    C  
    ATOM     17  OH  TYR B  28     110.135 163.913 139.789  1.00 54.45      B    O
    """.strip()
    MOCK_PDB = "\n".join(s.strip() for s in MOCK_PDB.split("\n"))

    @classmethod
    def predict(cls, sequence, *args, **kwargs):
        if (isinstance(sequence, str) and
                sequence == "THROW"):
            raise ValueError("Throwing as requested")
        result = prody.parsePDBStream(io.StringIO(cls.MOCK_PDB))
        assert result is not None
        return result
