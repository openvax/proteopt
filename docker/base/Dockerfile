FROM --platform=linux/x86_64 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as shrunk

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
WORKDIR /root

RUN apt-get update \
    && apt-get install -y wget git vim \
    && rm -rf /var/lib/apt/lists/* \
    && wget -nv \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda --version

# Make links
RUN ln -s /data/static/alphafold-params . \
    && ln -s /data/static/RFDesign . \
    && ln -s /data/static/AlphaFold . \
    && ln -s /data/static/openfold . \
    && ln -s /data/static/OmegaFold . \
    && ln -s /data/static/ProteinMPNN . \
    && ln -s /data/static/design-env/ miniconda3/envs/design-env \
    && mkdir -p /software/mlfold/alphafold-data \
    && ln -s /data/static/alphafold-params /software/mlfold/alphafold-data/params \
    && mkdir -p /data/static/omegafold_ckpt \
    && mkdir -p ~/.cache \
    && ln -s /data/static/omegafold_ckpt ~/.cache/omegafold_ckpt

# ****************************
FROM shrunk as complete
RUN mkdir -p /data/static

RUN cd /data/static \
    && mkdir -p alphafold-params \
    && wget -nv --progress=dot:giga --show-progress https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -O params.tar \
    && tar --extract --verbose --file=params.tar --directory=alphafold-params --preserve-permissions \
    && rm -f params.tar

# Note that we are using a different version of pytorch than recommended
# in RFDesign readme (they recommend pytorch=1.10.1)
RUN conda update -n base -c defaults conda \
  && conda config --set ssl_verify no \
  && conda init bash \
  && conda clean -afy

RUN rm miniconda3/envs/design-env \
    && conda create -n design-env \
        python=3.8 \
        pytorch=1.11 \
        dgl-cuda11.3 \
        cudatoolkit=11.3 \
        cuda-toolkit \
        numpy scipy requests packaging pip \
        -c "nvidia/label/cuda-11.3.1" -c pytorch -c dglteam  \
    && mv miniconda3/envs/design-env /data/static \
    && ln -s /data/static/design-env/ miniconda3/envs/design-env

RUN conda install -n design-env \
    pyg \
    openmm==7.5.1 \
    -c pyg -c conda-forge

RUN cd /data/static \
    && git clone https://github.com/deepmind/AlphaFold.git \
    && wget -q -P /data/static/AlphaFold/alphafold/common/ \
      https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN /root/miniconda3/envs/design-env/bin/pip install \
        https://github.com/openmm/pdbfixer/archive/refs/tags/v1.7.tar.gz \
        icecream==2.1.3 \
        lie_learn==0.0.1.post1 \
        opt_einsum==3.3.0 \
        e3nn==0.3.4 \
    && /root/miniconda3/envs/design-env/bin/pip install \
        "jax[cuda]==0.3.25" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && /root/miniconda3/envs/design-env/bin/pip install \
        dm-tree==0.1.6 \
        dm-haiku==0.0.9 \
        absl-py==1.0.0 \
        ml-collections==0.1.0 \
        tensorflow-gpu==2.11.0 \
        biopython==1.81 \
        pytorch-lightning==1.9.3 # For openfold \
    && conda clean -afy \
    && /root/miniconda3/envs/design-env/bin/pip install -e /data/static/AlphaFold \
    && /root/miniconda3/envs/design-env/bin/pip cache purge

RUN cd /data/static \
    && git clone --branch main https://github.com/timodonnell/RFDesign.git \
    && cd RFDesign \
    && git remote add upstream https://github.com/RosettaCommons/RFDesign.git \
    && wget -nv -P rfdesign/hallucination/weights/rf_Nov05 http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_last.pt \
    && wget -nv -P rfdesign/inpainting/weights/ http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_mix_epoch25.pt \
    && /root/miniconda3/envs/design-env/bin/pip install -e .

RUN cd /data/static \
    && git clone https://github.com/timodonnell/ProteinMPNN.git \
    && cd ProteinMPNN \
    && git remote add upstream https://github.com/dauparas/ProteinMPNN.git \
    && /root/miniconda3/envs/design-env/bin/pip install -e .

# OpenFold
RUN cd /data/static \
    && git clone https://github.com/timodonnell/openfold.git \
    && cd openfold \
    && git remote add upstream https://github.com/aqlaboratory/openfold.git \
    && /root/miniconda3/envs/design-env/bin/pip install -e . \
    && /root/miniconda3/envs/design-env/bin/pip cache purge \
    && mkdir -p /data/static/openfold-params/ \
    && bash scripts/download_openfold_params_huggingface.sh /data/static/openfold-params/

RUN mkdir -p /data/static/example-data \
    && wget -nv -P /data/static/example-data https://files.rcsb.org/download/7SL5.pdb
COPY container-files/test_protein.fa /data/static/example-data/

# We run OmegaFold to force it download the weights
RUN cd /data/static \
    && git clone https://github.com/timodonnell/OmegaFold \
    && cd OmegaFold \
    && git remote add upstream https://github.com/HeliXonProtein/OmegaFold \
    && /root/miniconda3/envs/design-env/bin/python setup.py install \
    && mkdir -p /tmp/omegafold_out \
    && /root/miniconda3/envs/design-env/bin/omegafold /data/static/example-data/test_protein.fa /tmp/omegafold_out \
    && ls -lh /tmp/omegafold_out \
    && rm -rf /tmp/omegafold_out

# Not sure why this is needed, but it seems to be:
RUN /root/miniconda3/envs/design-env/bin/pip install -e /data/static/AlphaFold

# Switch to the design-env on start:
RUN echo "conda activate design-env" >> ~/.bashrc
