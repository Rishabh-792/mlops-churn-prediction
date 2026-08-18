"""
deploy_sagemaker.py

Reference provisioning script for a SageMaker real-time endpoint.

STATUS: this path has never been run against a live AWS account. It is
committed as a design artifact showing how the trained ensemble would be
packaged and served, not as a system in operation. Nothing in CI executes it,
and the repository contains no AWS credentials.

To actually use it you would need: an AWS account with SageMaker permissions,
the ensemble packaged as model.tar.gz in S3, and `pip install .[aws]` plus the
`sagemaker` SDK.

Packaging the artifacts the way model_fn expects:

    cd artifacts && tar -czf model.tar.gz models/
    aws s3 cp model.tar.gz s3://<bucket>/models/churn-ensemble-v1.tar.gz
"""

from __future__ import annotations

import argparse


def deploy(
    s3_model_path: str,
    endpoint_name: str,
    instance_type: str = "ml.m5.large",
    instance_count: int = 1,
):
    """Creates or updates a SageMaker endpoint serving the ensemble."""
    # Imported lazily so the module stays importable (and lintable) without the
    # optional AWS dependency installed.
    import sagemaker
    from sagemaker.sklearn.model import SKLearnModel

    role = sagemaker.get_execution_role()

    model = SKLearnModel(
        model_data=s3_model_path,
        role=role,
        entry_point="inference.py",
        source_dir="deployment",
        framework_version="1.2-1",
        py_version="py3",
    )

    predictor = model.deploy(
        instance_type=instance_type,
        initial_instance_count=instance_count,
        endpoint_name=endpoint_name,
    )
    print(f"Endpoint live: {predictor.endpoint_name}")
    return predictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-model-path", required=True, help="s3://bucket/key/model.tar.gz")
    parser.add_argument("--endpoint-name", default="churn-ensemble")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    args = parser.parse_args()

    deploy(args.s3_model_path, args.endpoint_name, args.instance_type, args.instance_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
