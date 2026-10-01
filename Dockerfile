# syntax=docker/dockerfile:1

# Shared dependencies stage (cached: torch + pipeline libs downloaded once).
FROM python:3.11-slim-bookworm AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY requirements-pipeline.txt .
# CPU-only torch (the PyPI linux wheel pulls multi-GB NVIDIA CUDA dependencies).
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 \
        --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 -r requirements-pipeline.txt

# Builder stage: build the C++ core, run both test suites, and build the wheel.
FROM deps AS builder

COPY . .
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 scikit-build-core==0.11.4
RUN python -m pip wheel --no-cache-dir -w /wheels .
RUN python -m pip install --no-cache-dir /wheels/*.whl

RUN cmake -S . -B build && cmake --build build -j"$(nproc)" \
    && ctest --test-dir build --output-on-failure
RUN python -m pytest tests -q

# Final stage: run the full pipeline (fetch AV2 -> ... -> gate).
FROM deps AS final

ENV PDT_WORKERS=4

ARG TARGETARCH
RUN curl -fsSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-${TARGETARCH}.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd && chmod +x /usr/local/bin/s5cmd

WORKDIR /src
COPY --from=builder /src /src
COPY --from=builder /wheels /wheels
RUN rm -rf /src/build /src/build-cov
RUN python -m pip install --no-cache-dir /wheels/*.whl

CMD ["make", "pipeline"]
