# Sofima Deployment

07/27/23: Hardcoded read from S3 aind-open-data, and hardcoded writes to GCS sofima-test-bucket.
          To transition to S3 aind-open-data writes on tensorstore s3 support: https://github.com/google/tensorstore/pull/91

Define/Configure Inputs in `config.yml`.

Repository contains two (entrypoint, environment) pairs that correspond to independent capsules: 
- Registration:
  - Script: `register_dataset.py`
  - Environment: `registration_env.sh`

- Fusion:
  - Script: `fuse_and_multiscale.py`
  - Environment: `fuse_and_multiscale.sh`
