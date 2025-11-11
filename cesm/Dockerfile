# Use Ubuntu as the base image
FROM cesm_base:latest

# Add cert ccsm-cert (incommon signed)
COPY crt/svn-ccsm-cert.crt /usr/local/share/ca-certificates
RUN chmod 644 /usr/local/share/ca-certificates/svn-ccsm-cert.crt

# Add InCommon
COPY crt/incommon.crt /usr/local/share/ca-certificates
RUN chmod 644 /usr/local/share/ca-certificates/incommon.crt

# Update CA certificates
RUN update-ca-certificates

# SSL verification for SVN, proving difficult
RUN echo "[global]" > /etc/subversion/servers && \
    echo "ssl-authority-files = /etc/ssl/certs/ca-certificates.crt" >> /etc/subversion/servers && \
    echo "ssl-trust-default-ca = yes" >> /etc/subversion/servers && \
    echo "ssl-ignore-unknown-ca = no" >> /etc/subversion/servers

# Download CESM
# https://www.cesm.ucar.edu/models/releases
WORKDIR /opt
RUN wget https://github.com/ESCOMP/CESM/archive/refs/tags/release-cesm2.1.5.tar.gz && \
    tar -zxvf ./release-cesm2.1.5.tar.gz && \
    rm release-cesm2.1.5.tar.gz && \
    mv CESM-release-cesm2.1.5 cesm

# Install Dev, issue with gnu compiler broke in 2.1
#RUN wget https://github.com/ESCOMP/CESM/archive/refs/tags/release-cesm2.2.2.tar.gz && \
#    tar -zxvf ./release-cesm2.2.2.tar.gz && \
#    rm release-cesm2.2.2.tar.gz && \
#    mv CESM-release-cesm2.2.2 cesm

# Get Deps
WORKDIR /opt/cesm
RUN ./manage_externals/checkout_externals

# Set up Machine
# Set Machine Name
ENV MACH=gh200
ENV CESM_ROOT=/opt/cesm
ENV CIME_MACHINE=gh200
ENV USER=root

COPY config/config_compilers.xml /opt/cesm/cime/config/cesm/machines
COPY config/config_machines.xml /opt/cesm/cime/config/cesm/machines

WORKDIR /opt/cesm/cime/scripts
RUN mkdir -p /root/cesm/inputdata
ENV NETCDF_PATH=/usr/local

# Set some Flags
ENV LDFLAGS="-L/usr/local/lib -llapack -lblas"
ENV CXX_LDFLAGS="-L/usr/local/lib"

# Singularity 
RUN chmod -R 777 /opt/cesm

# Set up MPI (OpenMPI)
#RUN rm -rf /usr/local/mpi # This directory no longer exists
#RUN echo $(which mpiexec)
#RUN echo $(mpiexec --version) 
#RUN echo $(ls -al /usr/local/ompi)
#RUN echo $(ls -al /usr/local/ompi/bin)
ENV PATH="$PATH:/usr/local/ompi/bin" 
# OpenMPI is installed under /usr/local/ompi and not under /usr/local/mpi

# Nice to have
ENV PATH=/opt/cesm/cime/scripts:$PATH
WORKDIR /root

# For x86 Uncomment
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgfortran.so.5
