FROM python:3.6-stretch as base

ADD docker/deb-requirements.txt deb-requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
   DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" deb-requirements.txt) && \
   apt-get install -y --force-yes $DEBIAN_PACKAGES && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/* && \
   rm deb-requirements.txt

ADD docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt && rm requirements.txt

RUN git clone https://github.com/mblondel/soft-dtw.git
WORKDIR soft-dtw
RUN make cython
RUN python3 setup.py build
RUN python3 setup.py install

ADD docker/docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

COPY ntsa/ /ntsa
WORKDIR /ntsa

# Tensorboard
EXPOSE 6006

# Ipython
EXPOSE 8888

ENTRYPOINT ["./docker-entrypoint.sh"]
