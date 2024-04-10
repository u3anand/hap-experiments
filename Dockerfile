FROM ylxdzsw/hap:ae

WORKDIR /workspace

RUN apt-get update && apt-get install -y lsof

RUN apt-get install -y git

RUN git clone https://github.com/u3anand/hap-experiments.git

WORKDIR /workspace/hap-experiments

RUN pip install maturin
RUN PATH="/root/.cargo/bin:${PATH}" RUSTUP_TOOLCHAIN=nightly maturin build --release -o /workspace
RUN pip install /workspace/*.whl

RUN mkdir /profiler_data

ARG CACHEBUST=1

RUN git fetch --all && git reset --hard origin/main



