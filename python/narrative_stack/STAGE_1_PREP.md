# Stage 1 Model Setup

> **What is Stage 1?**
> Stage 1 trains a foundational autoencoder that learns latent embeddings of `(concept, unit, value)` triples. These embeddings form the atomic layer of a multi-stage modeling pipeline that ultimately encodes financial structure and behavior.

> **Note:** Run all steps from the `{project_root}/python` directory unless otherwise indicated.

---

## Dependencies and Environment

1. **Install Python dependencies**

   ```sh
   pip install -r requirements.txt
   ```

2. **Configure `.env` for database connectivity**

   ```sh
   cp .env.example .env
   ```

   Then update with your DB credentials.
   *Hint: `docker-compose.yml` may have helpful defaults.*

   > **Why is the DB needed?**
   > The database holds US GAAP taxonomy metadata: concept names, types, period/balance classifications, and labels. It does **not** store company-reported values. Those are loaded separately during preprocessing.

---

## Database Initialization

1. **Create the `us_gaap` schema**

   Use default settings:

   * Charset: `utf8mb4`
   * Collation: `utf8mb4_unicode_ci`

2. **Load the schema**

   ```sh
   mysql -u your_user -p us_gaap < db/sql/schema.sql
   ```

   This creates the following tables:

   * `us_gaap_concept`
   * `us_gaap_concept_type`
   * `us_gaap_balance_type`
   * `us_gaap_period_type`

---

## Taxonomy Ingestion and Training Data Preparation

1. **Download and extract the taxonomy**

   Open the notebook:

   ```
   notebooks/us_gaap_taxonomy_downloader.ipynb
   ```

   * Downloads the 2025 GAAP taxonomy ZIP

   * Extracts the Excel file

   * Converts the “Concepts” worksheet to CSV (`data/2025_GAAP_Concepts.csv`)

   > ⚠️ Update the year manually if using a newer taxonomy version.

2. **Ingest taxonomy into the DB**

   ```
   notebooks/ingest_us_gaap_concepts.ipynb
   ```

   This parses the CSV and populates the concept metadata tables.

3. **Generate Stage 1 training data**

   Open:

   ```
   notebooks/stage_1_preprocessing.ipynb
   ```

   This notebook:

   * Pulls raw facts from filings (not from the DB)
   * Joins each value with its corresponding taxonomy metadata
   * Applies scaling and structuring for training
