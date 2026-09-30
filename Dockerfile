FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd && chmod +x /usr/local/bin/s5cmd

WORKDIR /workspace/policy-divergence-triage
COPY . .

RUN pip install --no-cache-dir -e .

RUN cmake -S . -B build && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure

# Optional: install the data/ML pipeline dependencies for scenarios/*.py:
# RUN pip install --no-cache-dir -e ".[pipeline]"

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
