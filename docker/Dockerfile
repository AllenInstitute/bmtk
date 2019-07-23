FROM continuumio/anaconda3
MAINTAINER Kael Dai <kaeld@alleninstitute.org>

RUN apt-get update && apt-get install -y automake \
                                         libtool \
                                         build-essential \
                                         libncurses5-dev

ENV BUILD_DIR=/home/build
ENV HOME_DIR=/home/shared
ENV WORK_DIR=${HOME_DIR}/workspace

RUN mkdir -p ${BUILD_DIR}
RUN mkdir -p ${HOME_DIR}
RUN mkdir -p ${WORK_DIR}

RUN conda install -y numpy h5py lxml pandas matplotlib jsonschema scipy mpi4py cmake

# Install NEURON for BioNet
RUN conda install -y -c kaeldai neuron


### Install NEST for PointNet
ENV NEST_VER=2.12.0
ENV NEST_INSTALL_DIR=${BUILD_DIR}/nest/build
ENV PYTHON_ENV=python2.7

RUN cd ${BUILD_DIR} \
    conda install -y gsl; \
    wget --quiet https://github.com/nest/nest-simulator/releases/download/v${NEST_VER}/nest-${NEST_VER}.tar.gz -O nest.tar.gz; \
    tar xfz nest.tar.gz; \
    cd nest-${NEST_VER}; \
    mkdir build && cd build; \
    cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} -Dwith-mpi=ON -Dwith-gsl=ON -Dwith-python=ON -Dwith-ltdl=ON ..; \
    make; \
    make install

# Taken from /home/shared/nest/bin/nest_vars.sh, needed to run nest and pynest in jupyter
ENV NEST_DATA_DIR=${NEST_INSTALL_DIR}/share/nest
ENV NEST_DOC_DIR=${NEST_INSTALL_DIR}/share/doc/nest
ENV NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest
ENV NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/${PYTHON_ENV}/site-packages
ENV PYTHONPATH=${NEST_PYTHON_PREFIX}:${PYTHONPATH}
ENV PATH=${NEST_INSTALL_DIR}/bin:${PATH}


### Install DiPDE for PopNet
RUN conda install -y -c nicholasc dipde


### Install Tensorflow for MintNet
RUN conda install -y tensorflow

### Install AllenSDK (Not used by bmtk, but used by some notebooks to fetch cell-types files)
RUN pip install allensdk


### Install the bmtk
RUN cd ${BUILD_DIR}; \
    git clone https://github.com/AllenInstitute/bmtk.git; \
    cd bmtk; \
    python setup.py install

# Setup the examples and tutorials
RUN cd ${BUILD_DIR}/bmtk/docs; \
    cp -R examples ${HOME_DIR}; \
    cp -R tutorial ${HOME_DIR}

# Setup components directories for tutorials, including compiling neuron modfiles
RUN cd ${HOME_DIR}/tutorial; \
    cp -R ../examples/*_components .; \
    cd biophys_components/mechanisms; \
    nrnivmodl modfiles/

# Pre-compile mechanisms for BioNet examples
RUN cd ${HOME_DIR}/examples/biophys_components/mechanisms; \
    nrnivmodl modfiles/


# Create an entry point for running the image
COPY entry_script.sh ${BUILD_DIR}
RUN chmod +x ${BUILD_DIR}/entry_script.sh

ENTRYPOINT ["/home/build/entry_script.sh"]
