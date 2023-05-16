#!/bin/bash

# Run this inside the container to update repos
PIP=/root/miniconda3/envs/design-env/bin/pip
PYTHON=/root/miniconda3/envs/design-env/bin/python

$PIP install -r /root/proteopt/requirements.txt
$PIP install -e /root/proteopt

for i in ProteinMPNN RFDesign RFDiffusion
do
    cd ~/$i
    git fetch origin
    git reset --hard origin/main
done

cd ~/RFDiffusion/env/SE3Transformer
$PIP install --no-cache-dir -r requirements.txt
$PYTHON setup.py install

for i in ProteinMPNN RFDesign RFDiffusion
do
    cd ~/$i
    $PIP install -e .
done

$PIP install --upgrade biopython
