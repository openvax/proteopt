FROM timodonnell/proteopt-base-gpu:latest as base

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ='America/New York' apt-get install -y hhsuite kalign 

RUN /root/miniconda3/envs/design-env/bin/pip install \
    'ipywidgets<8' \
    pandas \
    seaborn \
    jupyterlab \
    prody \
    atomium \
    yabul \
    nglview \
    rmsd \
    ipdb \
    ipython \
    && conda clean -afy \
    && /root/miniconda3/envs/design-env/bin/pip cache purge

RUN mkdir -p /root/.jupyter
COPY jupyter_server_config.json /root/.jupyter/

EXPOSE 8888
CMD ["/root/miniconda3/envs/design-env/bin/jupyter" , "lab", "--ip=0.0.0.0", "--allow-root"]
