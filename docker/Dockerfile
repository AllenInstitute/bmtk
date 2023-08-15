# FROM continuumio/miniconda3:4.11.0
FROM continuumio/miniconda3:latest
MAINTAINER Kael Dai <kaeld@alleninstitute.org>

# RUN apt-get -y update && apt-get install -y automake \
#                                          libtool \
#                                          build-essential \
#                                          libncurses5-dev \
#                                          git \
#                                          cmake
#                                          # software-properties-common \
#                                          # dpkg-dev
#                                          # cmake
#                                          # libopenblas-dev
RUN apt-get -y update && apt-get install -y automake \
                                            build-essential \
                                            git


COPY environment.yml .
RUN conda env create -f environment.yml

# Make sure all further commands will run using bmtk environment
SHELL ["conda", "run", "-n", "bmtk", "/bin/bash", "-c"]

ENV BUILD_DIR=/home/build
ENV HOME_DIR=/home/shared
ENV WORK_DIR=${HOME_DIR}/workspace
ENV GIT_REPO=https://github.com/AllenInstitute/bmtk.git
ENV GIT_BRANCH=develop

RUN mkdir -p ${BUILD_DIR}
RUN mkdir -p ${HOME_DIR}
RUN mkdir -p ${WORK_DIR}

# #RUN conda update -n base -c defaults conda \
# #    && conda install python=3.6
# RUN conda install -y numpy h5py lxml pandas matplotlib jsonschema scipy sympy jupyter # cmake #  anaconda mpi4py mpich-mpicc


# ### Install NEURON for BioNet
# RUN pip install neuron


# ### Install NEST for PointNet
# #RUN conda install -c conda-forge nest-simulator==2.20.0
# #RUN add-apt-repository --enable-source ppa:nest-simulator/nest && \
# #    apt-get update && \
# #    apt-clean && \
# #    apt-get install -y nest
# ENV NEST_VER=2.20.0
# ENV NEST_INSTALL_DIR=${BUILD_DIR}/nest/build

# RUN cd ${BUILD_DIR} \
#     && conda install -y gsl cython \
#     && wget --quiet https://github.com/nest/nest-simulator/archive/v${NEST_VER}.tar.gz -O nest.tar.gz \
#     && tar xfz nest.tar.gz \
#     && cd nest-simulator-${NEST_VER} \
#     && mkdir build && cd build \
#     && cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} -Dwith-gsl=ON -Dwith-python=ON -Dwith-ltdl=ON .. \
#     && make \
#     && make install

# # Taken from /home/shared/nest/bin/nest_vars.sh, needed to run nest and pynest in jupyter
# ENV NEST_DATA_DIR "${NEST_INSTALL_DIR}/share/nest"
# ENV NEST_DOC_DIR "${NEST_INSTALL_DIR}/share/doc/nest"
# ENV NEST_MODULE_PATH "${NEST_INSTALL_DIR}/lib/nest"
# ENV NEST_PYTHON_PREFIX "${NEST_INSTALL_DIR}/lib/${PYTHON_ENV}/site-packages"
# ENV PYTHONPATH "${NEST_PYTHON_PREFIX}:${PYTHONPATH}"
# ENV PATH "${NEST_INSTALL_DIR}/bin:${PATH}"


#### Install DiPDE for PopNet
RUN cd ${BUILD_DIR}; \
    git clone https://github.com/AllenInstitute/dipde.git dipde; \
    pwd; \
    cd dipde; \
    pip install -e .;


#### Install the bmtk
RUN cd ${BUILD_DIR} \
    && git clone --branch ${GIT_BRANCH} ${GIT_REPO} \
    && cd bmtk \
    && python setup.py install


#### Setup the examples and tutorials
RUN cd ${BUILD_DIR}/bmtk; \
    cp -R examples ${HOME_DIR}; \
    cp -R docs/tutorial ${HOME_DIR}


# #### Add the tutorial for the 2022 workshop
# RUN cd ${HOME_DIR}/tutorial; \
#     git clone https://github.com/AllenInstitute/bmtk_workshop_2022.git

#### Setup components directories for tutorials, including compiling neuron modfiles
# RUN cd ${HOME_DIR}/tutorial \
#     && pwd \
#     && ls -l \
#     && cp -R ../../examples/*_components . \
#     && cd bio_components/mechanisms # \
#     # && nrnivmodl modfiles/

#### Pre-compile mechanisms for BioNet examples
RUN cd ${HOME_DIR}/examples/bio_components/mechanisms \
    && nrnivmodl modfiles/


#### Pre-compile mechanisms for BioNet examples
RUN cd ${HOME_DIR}/tutorial/modeling_tut_2021/components/mechanisms \
    && nrnivmodl modfiles/


#### Add the tutorial for the 2023 workshop
RUN cd ${HOME_DIR}; \
    git clone https://github.com/AllenInstitute/bmtk-workshop.git

#### Create an entry point for running the image
COPY entry_script.sh ${BUILD_DIR}
RUN chmod +x ${BUILD_DIR}/entry_script.sh
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bmtk", "/home/build/entry_script.sh"]
# ENTRYPOINT ["/home/build/entry_script.sh"]
