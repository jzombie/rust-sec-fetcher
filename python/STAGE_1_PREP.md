# Stage 1 Model Setup

TODO: What is Stage 1?

> Note: These steps should be performed within the context of the `{project_root}/python` directory, unless otherwise noted.

## Dependencies and Data

1. Install Python dependencies

    ```sh
    pip install
    ```

2. Configure `.env` file for DB connectivity. TODO: Document why the DB is needed.

    Copy `.env.example` to `.env` and configure accordingly.  
    Note: `docker-compose.yml` may provide configuration hints.

3. Create `us_gaap` schema in DB.

    Default Charset. 
    Default Collation.

4. Load DB schema backup (located in `db/sql/schema.sql`).

5. Ingest `US GAAP` taxonomies.

    TODO: For all included notebook steps, these notebooks may need to be modified to use the current year.

    1. Open `notebooks/us_gaap_taxonomy_downloader.ipynb` notebook.

        This will download the a zip file with an Excel spreadsheet, and then extract a CSV
        of the US GAAP concepts.

    2. Ingest into database

        Use `notebooks/ingest_us_gaap_concepts.ipynb` to ingest.

    3. Build training data

        Open `notebooks/stage_1_preprocessing.ipynb` 
