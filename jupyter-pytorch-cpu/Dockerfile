FROM 007ba7/master3.6:0.1

# install - requirements.txt
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
COPY --chown=jovyan:users uninstall.txt /tmp/uninstall.txt
RUN python3 -m pip uninstall -r /tmp/uninstall.txt --quiet --no-cache-dir
RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir \
 && rm -f /tmp/requirements.txt
RUN conda install pytorch cudatoolkit=10.1 -c pytorch
USER root
RUN apt install nvidia-driver-418
USER ${NB_UID}