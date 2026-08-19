FROM python:3.14-slim

WORKDIR /srv/churn

# Dependencies first so source changes do not bust the layer cache.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pyproject.toml ./
COPY configs/ configs/
COPY utils/ utils/
COPY pipelines/ pipelines/
COPY scripts/ scripts/
COPY deployment/ deployment/
COPY tests/ tests/

# Non-root runtime user.
RUN useradd --create-home runner && chown -R runner:runner /srv/churn
USER runner

# Default to the full pipeline; CI overrides this to run the test suite.
CMD ["sh", "-c", "python -m scripts.download_data && python -m pipelines.preprocessing_pipeline && python -m pipelines.training_pipeline && python -m pipelines.prediction_pipeline"]
