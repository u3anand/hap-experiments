FROM ylxdzsw/hap:ae

WORKDIR /workspace

RUN apt-get update && apt-get install -y lsof

RUN apt-get install -y git

RUN git clone https://github.com/u3anand/hap-experiments.git
