FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda:master-3dbc352f

# install - requirements.txt
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
# RUN conda install cv2 \
 && rm -f /tmp/requirements.txt 
USER root
RUN apt-get upgrade \
 && apt update && apt install -y libgl1-mesa-dev
USER ${NB_UID}