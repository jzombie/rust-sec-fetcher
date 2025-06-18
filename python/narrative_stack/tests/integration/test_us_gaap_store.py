import tempfile
import shutil
from models.pytorch.narrative_stack.common import UsGaapStore
from db import DbUsGaap

# . from utils.csv import generate_test_csv_dir


def test_ingestion_and_lookup():
    print("HELLO WORLD")

    # temp_dir = tempfile.mkdtemp()
    # try:
    #     # Create test CSVs in temp dir
    #     csv_dir = generate_test_csv_dir(temp_dir)

    #     # Create in-memory or stub DB object
    #     db = DbUsGaap()

    #     # Create store
    #     store_path = f"{temp_dir}/store.bin"
    #     store = UsGaapStore(store_path)

    #     # Ingest test data
    #     store.ingest_us_gaap_csvs(csv_dir, db)

    #     # Basic checks
    #     triplet_count = store.get_triplet_count()
    #     assert triplet_count > 0

    #     pair_count = store.get_pair_count()
    #     assert pair_count > 0

    #     # Lookup first i_cell
    #     result = store.lookup_by_index(0)
    #     assert "concept" in result
    #     assert "unscaled_value" in result

    #     # PCA model
    #     store.generate_pca_embeddings()
    #     pca_model = store.load_pca_model()
    #     assert pca_model is not None

    #     # Embedding retrieval
    #     embeddings, pairs = store.get_embedding_matrix()
    #     assert embeddings.shape[0] == len(pairs)

    # finally:
    #     shutil.rmtree(temp_dir)
