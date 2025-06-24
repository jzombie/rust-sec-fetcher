#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
source .venv/bin/activate

# DB container and network setup
DB_CONTAINER_NAME="us_gaap_test_db" # TODO: Rename w/ `TEST_` prefix
DB_NAME="us_gaap_test" # TODO: Rename w/ `TEST_` prefix
DB_NETWORK_NAME="narrative_stack_us_gaap_integration_test" # TODO: Rename w/ `TEST_` prefix
SCHEMA_FILE="tests/integration/assets/us_gaap_schema_2025.sql" # TODO: Rename w/ `TEST_` prefix

trap 'cleanup' EXIT

cleanup() {
  echo "Tearing down test container and network (if empty)..."
  docker rm -f "$DB_CONTAINER_NAME" 2>/dev/null || true

  # Remove only the test network if it's empty
  if docker network inspect "$DB_NETWORK_NAME" >/dev/null 2>&1; then
    if [[ "$(docker network inspect -f '{{json .Containers}}' "$DB_NETWORK_NAME")" == "{}" ]]; then
      docker network rm "$DB_NETWORK_NAME" || true
    fi
  fi
}

# Step 1: Start the test services (MySQL & SIMD R Drive)
# Note: Orbstack (Mac) *may* be referencing this wrong; if so, try with /usr/local/bin/docker
docker compose up --profile test -d

# Step 2: Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
until docker exec "$DB_CONTAINER_NAME" mysqladmin ping -h"127.0.0.1" --silent; do
  sleep 1
done
echo "MySQL is up."

# Step 3: Ensure database exists
echo "Ensuring database '$DB_NAME' exists..."
docker exec "$DB_CONTAINER_NAME" \
  mysql -uroot -ponlylocal -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\`;"

# Step 4: Apply schema
echo "Applying schema from $SCHEMA_FILE into database '$DB_NAME'..."
docker exec -i "$DB_CONTAINER_NAME" \
  mysql -uroot -ponlylocal "$DB_NAME" < "$SCHEMA_FILE"

# Step 5: Run integration test
echo "Running integration test..."
export PYTHONPATH=src
pytest -s -v tests/integration/test_us_gaap_store.py
