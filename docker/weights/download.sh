#!/bin/bash

set -e
set -x

mkdir -p alphafold-params
wget -nv -nc --progress=dot:giga --show-progress https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -O alphafold_params.tar || true
ls -lh alphafold_params.tar
tar --extract --verbose --file=alphafold_params.tar --directory=alphafold-params --preserve-permissions

mkdir -p rfdesign-params
pushd rfdesign-params
wget -nv -nc http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_last.pt || true
ls -lh BFF_last.pt
wget -nv -nc http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_mix_epoch25.pt  || true
ls -lh BFF_mix_epoch25.pt
popd

mkdir -p rfdiffusion-params
pushd rfdiffusion-params
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt  || true
wget -nv -nc http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt  || true
ls -lh Base_ckpt.pt Complex_base_ckpt.pt Complex_Fold_base_ckpt.pt \
  InpaintSeq_ckpt.pt InpaintSeq_Fold_ckpt.pt ActiveSite_ckpt.pt \
  Base_epoch8_ckpt.pt Complex_beta_ckpt.pt
popd

mkdir -p omegafold-params
pushd omegafold-params
wget -nv -nc --progress=dot:giga --show-progress \
  -O model.pt \
  https://helixon.s3.amazonaws.com/release1.pt || true
ls -lh model.pt
popd