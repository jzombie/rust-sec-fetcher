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

    TODO: This could especially be more automated.
    
    1. Download `GAAP Taxonomy` Excel sheet
    
        ```text
        # Note: The year will need to reflect the *current* year.

        https://www.fasb.org/page/detail?pageId=/projects/FASB-Taxonomies/2025-gaap-financial-reporting-taxonomy.html
        ```

        Look for:

        ```
        2025 GAAP Taxonomy (Excel Version) Taxonomy in a spreadsheet format to facilitate taxonomy review.
        ```

    2. Download the Excel version and save the `Concepts` tab (should be the default tab) to a CSV to `data/2025_GAAP_Concepts.csv`

        - Character Set: Unicode (UTF-8). 
        - Field delimiter: ,  
        - String delimiter: "  

        TODO: Document expected field rows, etc, for reference?


    n?. Download taxonomy hierarchies and extract ZIP file.
    n?. Extract taxonomy hierarchies to a CSV.

        - This is currently handled via `notebooks/us_gaap_statement_mapping.ipynb` and needs to be more automated.

        - The *current* output of this is `data/WITH_TAXONOMY_HIERARCHY_us_gaap_2025_with_all_statements_and_hierarchy.csv`

