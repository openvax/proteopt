import os
import tempfile
import argparse
import gc

import numpy
import numpy.testing
import prody

import yabul

import torch

from typing import Optional

from .common import args_from_function_signature

prody.confProDy(verbosity='none')


class OmegaFold(object):
    tool_name = "omegafold"

    def __init__(
            self,
            data_dir : str,
            device : str = "cuda",
            model_num : int = 1,
            subbatch_size : Optional[int] = None,
            num_cycle : int = 10):

        import omegafold
        import omegafold.config

        if model_num == 1:
            model_name = "model.pt"
        elif model_num == 2:
            model_name = "model2.pt"
        else:
            raise ValueError("Unsupported model num: %s" % model_num)

        weights = torch.load(
            os.path.join(data_dir, model_name), map_location='cpu')
        weights = weights.pop('model', weights)

        self.forward_config = argparse.Namespace(
            subbatch_size=subbatch_size,
            num_recycle=num_cycle,
        )

        self.model_num = model_num
        self.model = omegafold.OmegaFold(
            omegafold.config.make_config(int(self.model_num)))
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model.to(device)
        self.device = device

    config_args = args_from_function_signature(
        __init__, include=["data_dir", "device"])
    model_args = args_from_function_signature(
        __init__, exclude=list(config_args))

    def run_multiple(self, sequences, show_progress=False, items_per_request=None):
        import omegafold
        import omegafold.pipeline

        new_sequences = []
        for obj in sequences:
            if isinstance(obj, dict):
                assert list(obj) == ["sequence"]
                obj = obj["sequence"]
            new_sequences.append(obj)
        sequences = new_sequences

        results = {}
        temp_dir = None
        with torch.no_grad():
            try:
                temp_dir = tempfile.TemporaryDirectory("proteopt_omegafold")
                fasta_path = os.path.join(temp_dir.name, "input.fa")
                output_dir = os.path.join(temp_dir.name, "out")
                os.mkdir(output_dir)

                yabul.write_fasta(
                    fasta_path,
                    [("seq_%d" % i, sequences[i]) for i in range(len(sequences))])
                inputs = omegafold.pipeline.fasta2inputs(
                    fasta_path,
                    output_dir=output_dir,
                    device=self.device)

                if show_progress:
                    import tqdm
                    inputs = tqdm.tqdm(inputs, total=len(sequences))

                results = {}
                for i, (input_data, save_path) in enumerate(inputs):
                    output = self.model(
                        input_data,
                        predict_with_confidence=True,
                        fwd_cfg=self.forward_config
                    )
                    omegafold.pipeline.save_pdb(
                        pos14=output["final_atom_positions"],
                        b_factors=output["confidence"] * 100,
                        sequence=input_data[0]["p_msa"][0],
                        mask=input_data[0]["p_msa_mask"][0],
                        save_path=save_path,
                        model=0
                    )
                    handle = prody.parsePDB(save_path)
                    results[
                        os.path.basename(save_path).replace(".pdb", "")
                    ] = handle
                    torch.cuda.empty_cache()
                    gc.collect()

            finally:
                temp_dir.cleanup()

        ordered_results = [results["seq_%d" % i] for i in range(len(sequences))]
        for (sequence, handle) in zip(sequences, ordered_results):
            numpy.testing.assert_equal(handle.ca.getSequence(), sequence)
        return ordered_results

    def run(self, sequence : str):
        result, = self.run_multiple([sequence])
        return result

    run_args = args_from_function_signature(run)
