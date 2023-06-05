import collections
import logging
import tempfile
import time
import multiprocessing

from typing import Optional, List

import numpy
import random
import torch
import prody
import pandas

from .common import set_residue_data, args_from_function_signature

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from .scaffold_problem import (
    ScaffoldProblem, UnconstrainedSegment, ConstrainedSegment, ChainBreak)

# From RFDiffusion
def make_deterministic(seed=0):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

DEFAUT_CONFIG_YAML = """
inference:
  input_pdb: null
  num_designs: 1
  design_startnum: 0
  ckpt_override_path: null
  symmetry: null
  recenter: True
  radius: 10.0
  model_only_neighbors: False
  output_prefix: samples/design
  write_trajectory: True
  scaffold_guided: False
  model_runner: SelfConditioning
  cautious: True
  align_motif: True
  symmetric_self_cond: True
  final_step: 1
  deterministic: False
  trb_save_ckpt_path: null
  schedule_directory_path: null
  model_directory_path: null

contigmap:
  contigs: null
  inpaint_seq: null
  provide_seq: null
  length: null

model:
  n_extra_block: 4
  n_main_block: 32
  n_ref_block: 4
  d_msa: 256
  d_msa_full: 64
  d_pair: 128
  d_templ: 64
  n_head_msa: 8
  n_head_pair: 4
  n_head_templ: 4
  d_hidden: 32
  d_hidden_templ: 32
  p_drop: 0.15
  SE3_param_full:
    num_layers: 1
    num_channels: 32
    num_degrees: 2
    n_heads: 4
    div: 4
    l0_in_features: 8
    l0_out_features: 8
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 32
  SE3_param_topk:
    num_layers: 1
    num_channels: 32
    num_degrees: 2
    n_heads: 4
    div: 4
    l0_in_features: 64
    l0_out_features: 64
    l1_in_features: 3
    l1_out_features: 2
    num_edge_features: 64
  freeze_track_motif: False
  use_motif_timestep: False

diffuser:
  T: 50
  b_0: 1e-2
  b_T: 7e-2
  schedule_type: linear
  so3_type: igso3
  crd_scale: 0.25
  partial_T: null    
  so3_schedule_type: linear
  min_b: 1.5
  max_b: 2.5
  min_sigma: 0.02
  max_sigma: 1.5

denoiser:
  noise_scale_ca: 1
  final_noise_scale_ca: 1
  ca_noise_schedule_type: constant
  noise_scale_frame: 1
  final_noise_scale_frame: 1
  frame_noise_schedule_type: constant

ppi:
  hotspot_res: null

potentials:
  guiding_potentials: null 
  guide_scale: 10
  guide_decay: constant
  olig_inter_all : null
  olig_intra_all : null
  olig_custom_contact : null
  substrate: null

contig_settings:
  ref_idx: null
  hal_idx: null
  idx_rf: null
  inpaint_seq_tensor: null

preprocess:
  sidechain_input: False
  motif_sidechain_input: True
  d_t1d: 22
  d_t2d: 44
  prob_self_cond: 0.0
  str_self_cond: False
  predict_previous: False
  
logging:
  inputs: False

scaffoldguided:
  scaffoldguided: False
  target_pdb: False
  target_path: null
  scaffold_list: null
  scaffold_dir: null
  sampled_insertion: 0
  sampled_N: 0
  sampled_C: 0
  ss_mask: 0
  systematic: False
  target_ss: null
  target_adj: null
  mask_loops: True
  contig_crop: null
"""

DEFAULT_CONFIG = OmegaConf.create(DEFAUT_CONFIG_YAML)

def make_contigmap_from_problem(problem: ScaffoldProblem):
    contigs = []
    for (i, contig) in enumerate(problem.segments):
        if isinstance(contig, ConstrainedSegment):
            if contig.resindices is None:
                raise NotImplementedError("Must constrain structure")

            previous = None
            pieces = []
            for resindex in contig.resindices + 1:
                if previous is None:
                    # Start segment
                    pieces.append("%s%d-" % (contig.chain, resindex))
                elif resindex != previous + 1:
                    # End segment
                    pieces.append("%d" % previous)
                    pieces.append("/%s%d-" % (contig.chain, resindex))
                previous = resindex
            # End final segment
            pieces.append("%d" % previous)
            contigs.append("".join(pieces))
        elif isinstance(contig, UnconstrainedSegment):
            contigs.append(f"{contig.length}-{contig.length}")
        elif isinstance(contig, ChainBreak):
            contigs.append("0 ")
        else:
            raise ValueError(contig)

    result = "/".join(contigs).replace("/0 /", "/0 ")
    return result


class RFDiffusionMotif(object):
    tool_name = "rfdiffusion_motif"

    def __init__(self, models_dir : str, num_processes : int = 0, num_timesteps : Optional[int] = None):
        self.num_processes = num_processes
        self.conf = DEFAULT_CONFIG.copy()
        self.conf.inference.model_directory_path = models_dir
        if num_timesteps is not None:
            self.conf.diffuser.T = num_timesteps

    config_args = args_from_function_signature(
        __init__, include=["models_dir", "num_processes"])
    model_args = args_from_function_signature(
        __init__, exclude=list(config_args))

    def run_multiple(
            self,
            problems : List[ScaffoldProblem],
            show_progress=False,
            items_per_request=None):

        if self.num_processes != 0:
            # Use multiprocessing (experimental)
            context = multiprocessing.get_context("spawn")
            dicts = [
                p if isinstance(p, dict) else {"problem": p} for p in problems
            ]
            with context.Pool(processes=self.num_processes) as pool:
                return pool.map(self.run_from_dict, dicts)
        else:
            results = []
            if show_progress:
                import tqdm
                problems_iterator = tqdm.tqdm(problems)
            else:
                problems_iterator = iter(problems)

            for problem in problems_iterator:
                if isinstance(problem, dict):
                    result = self.run(**problem)
                else:
                    result = self.run(problem)
                results.append(result)
            return results

    def run_from_dict(self, d):
        return self.run(**d)

    def run(self, problem : ScaffoldProblem, num : int = 1):
        import rfdiffusion
        import rfdiffusion.inference.utils

        if self.conf.inference.deterministic:
            make_deterministic(0)

        config = self.conf.copy()
        contigmap = make_contigmap_from_problem(problem)
        logging.info("rfdiffusion contigmap: %s", contigmap)
        config.contigmap.contigs = [contigmap]

        with tempfile.NamedTemporaryFile(suffix=".pdb") as input_pdb:
            input_structure = problem.get_structure()
            prody.writePDB(input_pdb.name, input_structure)
            config.inference.input_pdb = input_pdb.name

            sampler = rfdiffusion.inference.utils.sampler_selector(config)

            results = []
            for design_num in range(num):
                start_time = time.time()
                x_init, seq_init = sampler.sample_init()
                denoised_xyz_stack = []
                px0_xyz_stack = []
                seq_stack = []
                plddt_stack = []

                x_t = torch.clone(x_init)
                seq_t = torch.clone(seq_init)
                # Loop over number of reverse diffusion time steps.
                for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
                    px0, x_t, seq_t, plddt = sampler.sample_step(
                        t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
                    )
                    px0_xyz_stack.append(px0)
                    denoised_xyz_stack.append(x_t)
                    seq_stack.append(seq_t)
                    plddt_stack.append(plddt[0])  # remove singleton leading dimension

                # Flip order for better visualization in pymol
                denoised_xyz_stack = torch.stack(denoised_xyz_stack)
                denoised_xyz_stack = torch.flip(
                    denoised_xyz_stack,
                    [
                        0,
                    ],
                )

                plddt_stack = torch.stack(plddt_stack)

                # Output glycines, except for motif region
                final_seq = torch.where(
                    torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
                )  # 7 is glycine

                bfacts = torch.ones_like(final_seq.squeeze())
                # make bfact=0 for diffused coordinates
                bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
                # pX0 last step

                # Now don't output sidechains
                with tempfile.NamedTemporaryFile(suffix=".pdb") as fd:
                    rfdiffusion.util.writepdb(
                        fd.name,
                        denoised_xyz_stack[0, :, :4],
                        final_seq,
                        sampler.binderlen,
                        chain_idx=sampler.chain_idx,
                        bfacts=bfacts,
                    )
                    pdb_result = prody.parsePDB(fd.name)
                    problem.annotate_solution(pdb_result)
                    problem.check_solution_sequence_is_valid(
                        pdb_result.ca.getSequence(), ipdb_on_error=False)

                results.append(collections.OrderedDict([
                    ("design_num", design_num),
                    ("structure", pdb_result),
                    ("plddt", plddt_stack.cpu().numpy()),
                    ("time", time.time() - start_time),
                    ("device",
                     torch.cuda.get_device_name(torch.cuda.current_device())
                     if torch.cuda.is_available() else "CPU"),
                    ("contig_mappings", sampler.contig_map.get_mappings()),
                ]))
            results = pandas.DataFrame(results)
        return results
    run_args = args_from_function_signature(run)
