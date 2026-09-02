# End-to-End MLOps Pipeline

This repository contains a complete, production-grade MLOps pipeline.

## Getting Started

1. **Train the Model:**
   ```bash
   pip install -r requirements.txt
   python train.py
   ```
   This will generate a `models/model.pkl` file using MLflow tracking.

2. **Run Locally (with Monitoring):**
   ```bash
   docker-compose up --build -d
   ```
   * API: `http://localhost:8000/docs`
   * Prometheus: `http://localhost:9090`
   * Grafana: `http://localhost:3000`

3. **Use the Interactive CLI:**
   ```bash
   python mlops_cli.py
   ```
