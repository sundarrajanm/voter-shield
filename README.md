# VoterShield

VoterShield is a Python-based data processing pipeline designed to convert **scanned electoral roll PDFs** into **structured, analyzable voter data** with high accuracy and reproducibility.

The project focuses on:
- Deterministic OCR parsing
- Strong regression guarantees
- Production-grade quality gates
- Horizontal scalability via containerization

---

## 📌 Problem Statement

Electoral rolls are typically published as scanned PDFs containing:
- Thousands of voter records per document
- Fixed visual layouts but noisy OCR output
- High sensitivity to parsing errors

Manual extraction is error-prone and non-scalable.  
VoterShield provides a **repeatable, testable, and scalable** approach to this problem.

---

## 🧠 Core Design Principles

- **Determinism over heuristics**  
  OCR parsing relies on explicit markers (`VOTER_END`) instead of positional guessing.

- **Golden-file regression testing**  
  Every change is validated against a known-correct baseline.

- **Low-noise codebase**  
  Strict linting, formatting, and unused-code elimination.

- **Embarrassingly parallel architecture**  
  One electoral booth → one container → horizontal scaling.

---

## 🏗️ High-Level Architecture

```
PDF (Scanned)
↓
PDF → Image Conversion
↓
OCR (Text Extraction)
↓
Voter Block Splitting (VOTER_END-based)
↓
Field Extraction & Normalization
↓
CSV Output
```


Each step is isolated and testable.

---

## 📂 Repository Structure

```
├── main.py # Pipeline entry point
├── requirements.txt # Runtime dependencies
├── requirements-dev.txt # Development & quality tools
├── scripts/
│ ├── quality.sh # Lint, format, and test gate
│ └── run-docker.sh # Optional Docker helper
├── tests/
│ ├── fixtures/ # Golden PDFs and CSVs
│ └── test_regression.py # Regression test suite
├── Dockerfile # Calibration-grade container image
├── Makefile # Primary developer interface
├── pyproject.toml # Tool configuration
└── README.md
```


---

## 🚀 Getting Started (Local)

### Prerequisites
- Python 3.10+
- Docker (for container calibration)
- GNU Make

## 🧰 Common Tasks (Makefile)

The **Makefile is the recommended interface** for working with this project.

### Install dependencies

```bash
make setup
```

### Run quality checks (mandatory)
```
make check-quality
```

This enforces:
* Ruff linting + autofix
* Black formatting
* Pytest regression tests

### Run the pipeline locally

```
make run
```

### Run inside Docker (development / calibration)
```
make run-dev-docker
```

This runs the pipeline:

* Inside the Docker container
* With constrained CPU and memory
* Using a bind-mounted codebase for fast iteration

This mode is ideal for performance tuning and memory calibration.

## 🧪 Regression Testing
VoterShield uses golden-file regression testing to prevent subtle OCR regressions.

* Known PDFs are processed
* Generated CSVs are compared field-by-field
* Only actual differences are reported

Tests can also be run directly:

```
pytest -q -ra --disable-warnings
```

## 🐳 Docker & Scaling Model

The Docker image is designed for calibration and runtime parity.

Key assumptions:
* 1 booth per container
* 1 CPU-bound execution
* No shared state
* No multi-threading inside the container

This design aligns naturally with AWS Fargate and similar platforms.

## 📊 Performance Calibration

Performance is evaluated using the Docker container as the baseline, focusing on:

* Booth-level wall-clock runtime
* Peak memory usage
* CPU saturation
* Cost-per-booth in cloud environments

The goal is predictable horizontal scaling, not vertical optimization.

## 🧩 Non-Goals

This project intentionally does not include:

* Real-time processing
* Database persistence
* Analytics or visualization layers

Downstream systems are expected to consume the generated CSV outputs.

## 🤝 Contribution Guidelines
* Always run `make check-quality` before submitting changes
* Avoid heuristic parsing without updating regression fixtures
* Preserve determinism and test coverage
* Prefer clarity over cleverness


