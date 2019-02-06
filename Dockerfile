FROM tensorflow/tensorflow:1.12.0-gpu-py3

VOLUME /opt
WORKDIR /opt

RUN apt-get update
RUN apt-get install -y \
    vim \
    tmux \
    screen \
    git \
    curl

# python packages
RUN pip install --upgrade pip
RUN pip install numpy==1.16.0
RUN pip install \
    pandas \
    scikit-learn \
    tensorflow-gpu \
    keras \
    dotmap \
    torch \
    torchvision \
    seaborn \
    gensim \
    nltk \
    PyYAML \
    python-xmp-toolkit \
    num2words \
    ekphrasis \
    jupyter \
    jupyterlab \
    beautifulsoup4 \
    html5lib \
    h5py \
    Keras \
    wordcloud \
    librosa \
    jupyter_contrib_nbextensions

RUN jupyter contrib nbextension install --user
