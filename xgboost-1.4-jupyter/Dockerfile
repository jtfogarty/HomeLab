FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch:master-3dbc352f

#USER root

# install gcc
#RUN export DEBIAN_FRONTEND=noninteractive \
#&& curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
#&& echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
#&& apt-get -yq update \
#&& apt-get -y install --no-install-recommends \
#   build-essential \
#&& apt-get clean \
#&& rm -rf /var/lib/apt/lists/*

#USER ${NB_UID}

# install - requirements.txt
COPY --chown=jovyan:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --no-cache-dir \
 && rm -f /tmp/requirements.txt
