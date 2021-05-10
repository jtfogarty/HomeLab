FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda:master-3dbc352f

# install - requirements.txt
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --quiet --no-cache-dir \
# RUN conda install cv2 \
 && rm -f /tmp/requirements.txt 
RUN apt-get upgrade \
 && apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx