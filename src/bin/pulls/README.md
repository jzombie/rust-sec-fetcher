Direct files to Parquet example

```python
import duckdb

# 2GB in bytes
CHUNK_SIZE = 2 * 1024 * 1024 * 1024 

duckdb.sql(f"""
    COPY (
        SELECT 
            regexp_extract(filename, '([^/]+)/([^/]+)\.htm', 1) AS ticker_symbol,
            regexp_extract(filename, '([^/]+)/([^/]+)\.htm', 2) AS accession_number,
            column0 AS raw_text
        FROM read_csv('data/*/*.htm', 
                      columns={{'column0': 'VARCHAR'}}, 
                      delim=None, 
                      header=False, 
                      filename=True)
    ) TO 'hf_upload_folder' (
        FORMAT 'PARQUET',
        FILE_TEMPLATE 'train-{{uuid}}', 
        MAX_FILE_SIZE {CHUNK_SIZE},
        COMPRESSION 'ZSTD' -- ZSTD is best for text balance of speed/size
    );
""")
```
