# Security Policy

## Supported versions

This is a portfolio/reference project. Security fixes are applied to the
`main` branch only.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
[GitHub Security Advisories](https://github.com/Rishabh-792/mlops-churn-prediction/security/advisories/new)
rather than opening a public issue. I aim to acknowledge reports within 7 days.

## Data and credentials

- No credential is required to run, test, or train this project. CI executes
  the entire pipeline with no secrets configured.
- The training dataset is public (IBM Telco Customer Churn) and is downloaded
  at build time rather than committed, verified against a pinned SHA-256.
- No customer, employer, or third-party data appears in this repository. The
  test suite runs on a synthetic frame generated at test time.

If you believe you have found a committed secret, please report it privately
using the link above.
