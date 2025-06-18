import os
import tempfile
from models.pytorch.narrative_stack.common import UsGaapStore
from db import DbUsGaap
from simd_r_drive import DataStore

# Get the directory containing the script (do not change)
script_dir = os.path.dirname(os.path.abspath(__file__))


def test_ingestion_and_lookup():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSVs in temp dir
        csv_dir = os.path.join(script_dir, "assets/truncated_csvs")

        # Create in-memory or stub DB object
        db = DbUsGaap()

        # Create store
        store_path = f"{temp_dir}/store.bin"
        data_store = DataStore(store_path)

        us_gaap_store = UsGaapStore(data_store)

        # Ingest test data
        us_gaap_store.ingest_us_gaap_csvs(csv_dir, db)

        # Basic checks
        triplet_count = us_gaap_store.get_triplet_count()
        assert triplet_count > 0

        # PCA model
        us_gaap_store.generate_pca_embeddings()
        pca_model = us_gaap_store.load_pca_model()
        assert pca_model is not None

        pair_count = us_gaap_store.get_pair_count()
        assert pair_count > 0

        # Lookup first i_cell
        result = us_gaap_store.lookup_by_index(0)
        assert "concept" in result
        assert "unscaled_value" in result

        # Embedding retrieval
        embeddings, pairs = us_gaap_store.get_embedding_matrix()
        assert embeddings.shape[0] == len(pairs)
