FROM ylxdzsw/hap:ae

WORKDIR /root

RUN apt-get update && apt-get install -y lsof

RUN apt-get install -y git

ARG CACHEBUST=1

RUN mkdir /root/hap-experiments
# RUN git clone https://github.com/u3anand/hap-experiments.git

WORKDIR /root/hap-experiments

RUN mkdir /profiler_data

COPY . /root/hap-experiments
COPY ./network/ssh_config /root/.ssh/config
RUN /usr/sbin/sshd

# RUN cp /root/hap/hap.so /root/hap-experiments
# RUN pip install maturin
# RUN PATH="/root/.cargo/bin:${PATH}" RUSTUP_TOOLCHAIN=nightly maturin build --release -o /workspace
# RUN pip install /workspace/*.whl
RUN PATH="/root/.cargo/bin:${PATH}" cargo +nightly build --release
RUN cp target/release/libhap.so /root/hap-experiments/hap.so

EXPOSE 3922

