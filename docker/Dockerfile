FROM --platform=linux/x86_64 nvidia/cuda:11.8.0-devel-ubuntu22.04 as minimal

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
WORKDIR /root

# Base packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ='America/New York' apt-get install -y \
      hhsuite \
      kalign \
      wget \
      git \
      vim \
    && rm -rf /var/lib/apt/lists/* \
    && wget -nv \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda --version

# Setup cudnn
RUN wget -nv --progress=dot:giga --show-progress \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb \
    && dpkg -i libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb \
    && rm libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb

# Make links
RUN ln -s /data/static/alphafold-params . \
    && ln -s /data/static/RFDesign . \
    && ln -s /data/static/RFDiffusion . \
    && ln -s /data/static/rfdiffusion-params . \
    && ln -s /data/static/AlphaFold . \
    && ln -s /data/static/openfold . \
    && ln -s /data/static/openfold-params . \
    && ln -s /data/static/proteopt . \
    && ln -s /data/static/OmegaFold . \
    && ln -s /data/static/ProteinMPNN . \
    && ln -s /data/static/ColabDesign . \
    && ln -s /data/static/design-env/ miniconda3/envs/design-env \
    && mkdir -p /software/mlfold/alphafold-data \
    && ln -s /data/static/alphafold-params /software/mlfold/alphafold-data/params \
    && mkdir -p /data/static/omegafold_ckpt \
    && mkdir -p ~/.cache \
    && ln -s /data/static/omegafold_ckpt ~/.cache/omegafold_ckpt

# Was getting some errors from wandb without this:
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# ****************************
FROM minimal as base
RUN mkdir -p /data/static

##### WEIGHTS DOWNLOADS #####
# We try to do as many weights downloads as possible initially so tweaking
# e.g. python versions later doesn't trigger a redownload.
# Alphafold weights
RUN cd /data/static \
    && mkdir -p alphafold-params \
    && wget -nv --progress=dot:giga --show-progress https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar -O params.tar \
    && tar --extract --verbose --file=params.tar --directory=alphafold-params --preserve-permissions \
    && rm -f params.tar

# RFDesign weights
RUN mkdir /data/static/rfdesign-params \
    && cd /data/static/rfdesign-params \
    && wget -nv http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_last.pt \
    && wget -nv http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_mix_epoch25.pt

# RFDiffusion weights
RUN mkdir /data/static/rfdiffusion-params \
    && cd /data/static/rfdiffusion-params \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt \
    && wget -nv http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt

# OpenFold
RUN cd /data/static \
    && git clone https://github.com/timodonnell/openfold.git \
    && cd openfold \
    && git remote add upstream https://github.com/aqlaboratory/openfold.git \
    && mkdir -p /data/static/openfold-params/ \
    && bash scripts/download_openfold_params_huggingface.sh /data/static/openfold-params/

# Omegafold
RUN wget -nv --progress=dot:giga --show-progress \
    -O /root/.cache/omegafold_ckpt/model.pt \
    https://helixon.s3.amazonaws.com/release1.pt
RUN wget -nv --progress=dot:giga --show-progress \
    -O /root/.cache/omegafold_ckpt/model2.pt \
    https://helixon.s3.amazonaws.com/release2.pt

##### ENDS WEIGHTS DOWNLOADS #####

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
        dgl-cuda11.6 \
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

# conda environment setup
RUN /root/miniconda3/envs/design-env/bin/pip install \
        https://github.com/openmm/pdbfixer/archive/refs/tags/v1.7.tar.gz \
        icecream==2.1.3 \
        lie_learn==0.0.1.post1 \
        opt_einsum==3.3.0 \
        e3nn==0.3.4 \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
    && /root/miniconda3/envs/design-env/bin/pip install \
        "jax[cuda]==0.4.8" \
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

# Alphafold install
RUN cd /data/static \
    && git clone https://github.com/deepmind/AlphaFold.git \
    && wget -q -P /data/static/AlphaFold/alphafold/common/ \
      https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
    && /root/miniconda3/envs/design-env/bin/pip install -e ~/AlphaFold

# ColabDesign
RUN cd /data/static \
    && git clone --branch main https://github.com/sokrypton/ColabDesign.git \
    && cd ColabDesign \
    && /root/miniconda3/envs/design-env/bin/pip install -e ~/ColabDesign

# RFDesign
RUN cd /data/static \
    && git clone --branch main https://github.com/timodonnell/RFDesign.git \
    && cd RFDesign \
    && git remote add upstream https://github.com/RosettaCommons/RFDesign.git \
    && /root/miniconda3/envs/design-env/bin/pip install -e ~/RFDesign \
    && mkdir -p rfdesign/hallucination/weights/rf_Nov05/ \
    && ln -s /data/static/rfdesign-params/BFF_last.pt rfdesign/hallucination/weights/rf_Nov05/BFF_last.pt \
    && mkdir -p rfdesign/inpainting/weights/ \
    && ln -s /data/static/rfdesign-params/BFF_mix_epoch25.pt rfdesign/inpainting/weights/BFF_mix_epoch25.pt \
    && ls -lh rfdesign/hallucination/weights/rf_Nov05/BFF_last.pt\
    && ls -lh rfdesign/inpainting/weights/BFF_mix_epoch25.pt

# ProteinMPNN
RUN cd /data/static \
    && git clone https://github.com/timodonnell/ProteinMPNN.git \
    && cd ProteinMPNN \
    && git remote add upstream https://github.com/dauparas/ProteinMPNN.git \
    && /root/miniconda3/envs/design-env/bin/pip install -e ~/ProteinMPNN

# OpenFold
RUN cd /data/static/openfold \
    && git pull origin main \
    && /root/miniconda3/envs/design-env/bin/pip install -e ~/openfold

# RFDiffusion
RUN cd /data/static \
    && git clone --branch main https://github.com/timodonnell/RFDiffusion.git \
    && cd RFDiffusion \
    && git remote add upstream https://github.com/RosettaCommons/RFDiffusion.git \
    && ln -s /data/static/rfdiffusion-params models

RUN cd /data/static/RFDiffusion/env/SE3Transformer \
    && /root/miniconda3/envs/design-env/bin/pip install --no-cache-dir -r requirements.txt \
    && python setup.py install \
    && cd /data/static/RFDiffusion \
    && /root/miniconda3/envs/design-env/bin/pip install -e /data/static/RFDiffusion --no-deps

# We run OmegaFold to test that it has the weights.
RUN mkdir -p /data/static/example-data \
    && wget -nv -P /data/static/example-data https://files.rcsb.org/download/7SL5.pdb \
    && ls -lh /root/.cache/omegafold_ckpt/model.pt

COPY container-files/test_protein.fa /data/static/example-data/
RUN cd /data/static \
    && git clone https://github.com/timodonnell/OmegaFold \
    && cd OmegaFold \
    && git remote add upstream https://github.com/HeliXonProtein/OmegaFold \
    && /root/miniconda3/envs/design-env/bin/python setup.py install \
    && mkdir -p /tmp/omegafold_out \
    && /root/miniconda3/envs/design-env/bin/omegafold /data/static/example-data/test_protein.fa /tmp/omegafold_out \
    && ls -lh /tmp/omegafold_out \
    && rm -rf /tmp/omegafold_out

# Clone proteopt repo
RUN cd /data/static \
    && git clone https://github.com/openvax/proteopt.git \
    && cd proteopt \
    && /root/miniconda3/envs/design-env/bin/pip install -r requirements.txt \
    && /root/miniconda3/envs/design-env/bin/pip install -e .

# Switch to the design-env on start:
RUN echo "conda activate design-env" >> ~/.bashrc


# ****************************
# Jupyter setup for interactive usage
FROM base as complete

RUN /root/miniconda3/envs/design-env/bin/pip install \
    'ipywidgets<8' \
    pandas \
    seaborn \
    jupyterlab \
    prody \
    yabul \
    nglview \
    rmsd \
    ipdb \
    ipython \
    optuna \
    optuna-dashboard \
    && conda clean -afy \
    && /root/miniconda3/envs/design-env/bin/pip cache purge

RUN mkdir -p /root/.jupyter
COPY container-files/jupyter_server_config.json /root/.jupyter/

RUN cd /root/proteopt/docker && bash update.sh

EXPOSE 8888 9999
CMD ["/root/miniconda3/envs/design-env/bin/jupyter" , "lab", "--ip=0.0.0.0", "--allow-root"]
