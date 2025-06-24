# Note: `git lfs` must be enabled and updated. See corresponding
# `.github/workflows/us-gaap-store-integration-test.yml` for local setup.

#!/usr/bin/env bash
set -euo pipefail

# Use an isolated Compose project just for this test run
PROJECT_NAME="us_gaap_it"          # anything unique
COMPOSE="docker compose -p $PROJECT_NAME --profile test"

source .venv/bin/activate
uv pip install -e . --group dev

trap 'cleanup' EXIT
cleanup() {
  echo "Tearing down test containers..."
  # This affects only containers in project $PROJECT_NAME
  $COMPOSE down --volumes --remove-orphans
}

# ------------------------------------------------------------------
# bring up ONLY the services tagged with `profiles: [test]`
$COMPOSE up -d   # starts db_test + simd_r_drive_ws_server_test

echo "Waiting for MySQL..."
until docker exec us_gaap_test_db mysqladmin ping -h127.0.0.1 --silent; do
  sleep 1
done

docker exec us_gaap_test_db \
  mysql -uroot -ponlylocal -e 'CREATE DATABASE IF NOT EXISTS `us_gaap_test`;'

docker exec -i us_gaap_test_db \
  mysql -uroot -ponlylocal us_gaap_test \
  < tests/integration/assets/us_gaap_schema_2025.sql

export PYTHONPATH=src
pytest -s -v tests/integration/test_us_gaap_store.py
