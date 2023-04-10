#!/bin/bash

# Run this inside the container to update repos
PIP=/root/miniconda3/envs/design-env/bin/pip

$PIP install -r /root/proteopt/requirements.txt
$PIP install -e /root/proteopt

for i in ProteinMPNN RFDesign RFDiffusion
do
    cd ~/$i
    git fetch origin
    git reset --hard origin/main
    $PIP install -e .
    cd ..
done
$PIP install --upgrade biopython