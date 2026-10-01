PYTHON ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)
BUILD_DIR ?= build

.PHONY: build test test-cpp test-python coverage scenarios train shadow metrics cluster \
        review ab gate pipeline fixture-pipeline clean

build:
	cmake -S . -B $(BUILD_DIR)
	cmake --build $(BUILD_DIR) -j

test: test-cpp test-python

test-cpp: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-python:
	$(PYTHON) -m pytest tests -q

coverage:
	cmake -S . -B build-cov -DPDT_COVERAGE=ON -DPDT_BUILD_PYTHON=OFF
	cmake --build build-cov -j
	ctest --test-dir build-cov --output-on-failure
	gcovr build-cov --root . --filter cpp/src --filter cpp/include --txt
	$(PYTHON) -m coverage run --source=pdt -m pytest tests -q
	$(PYTHON) -m coverage report -m

scenarios:
	$(PYTHON) scenarios/fetch_av2.py
	$(PYTHON) scenarios/convert_av2.py

train:
	$(PYTHON) -m pdt.train_policy --scenarios scenarios/logs.jsonl

shadow:
	$(PYTHON) -m pdt.harness --scenarios scenarios/logs.jsonl --split held_out

metrics: shadow

cluster:
	$(PYTHON) -m pdt.cluster --divergence artifacts/divergence.parquet

review:
	$(PYTHON) -m pdt.review --non-interactive --labels labels/cluster_labels.yaml

ab: shadow
	$(PYTHON) -m pdt.ab --scenarios scenarios/logs.jsonl \
		--baseline-divergence artifacts/divergence.parquet --baseline-events artifacts/override_events.parquet

gate:
	$(PYTHON) -m pdt.gate --report artifacts/ab_report.json --config gate_config.yaml

pipeline: scenarios train shadow cluster review ab gate

fixture-pipeline:
	mkdir -p artifacts/fixture-pipeline
	$(PYTHON) -m pdt.train_policy --scenarios tests/fixtures/scenarios.jsonl --epochs 2 \
		--out artifacts/fixture-pipeline/policy.pt --log artifacts/fixture-pipeline/train_log.jsonl
	$(PYTHON) -m pdt.harness --scenarios tests/fixtures/scenarios.jsonl --split all \
		--checkpoint artifacts/fixture-pipeline/policy.pt \
		--out artifacts/fixture-pipeline/trajectories.parquet \
		--metrics-out artifacts/fixture-pipeline/divergence.parquet \
		--events-out artifacts/fixture-pipeline/override_events.parquet
	$(PYTHON) -m pdt.cluster --divergence artifacts/fixture-pipeline/divergence.parquet \
		--threshold 0.0 --out-dir artifacts/fixture-pipeline
	$(PYTHON) scripts/write_fixture_labels.py
	$(PYTHON) -m pdt.ab --scenarios tests/fixtures/scenarios.jsonl --min-target-size 1 \
		--checkpoint artifacts/fixture-pipeline/policy.pt \
		--model artifacts/fixture-pipeline/cluster_model.joblib \
		--labels labels/fixture_labels.yaml --out-dir artifacts/fixture-pipeline \
		--baseline-divergence artifacts/fixture-pipeline/divergence.parquet \
		--baseline-events artifacts/fixture-pipeline/override_events.parquet
	$(PYTHON) -m pdt.gate --report artifacts/fixture-pipeline/ab_report.json --config gate_config.yaml

clean:
	rm -rf $(BUILD_DIR) build-cov artifacts scenarios/av2_raw scenarios/logs.jsonl
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage
