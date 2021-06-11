#FROM 007ba7/master3.6:0.1
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:master-1831e436

# install - requirements.txt
COPY --chown=jovyan:users uninstall.txt /tmp/uninstall.txt
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip uninstall -r /tmp/uninstall.txt --quiet --no-cache-dir
RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir \
 && rm -f /tmp/requirements.txt
RUN conda install pytorch cudatoolkit=10.1 -c pytorch