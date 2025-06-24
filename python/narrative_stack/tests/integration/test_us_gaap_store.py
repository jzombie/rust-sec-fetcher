import os
from models.pytorch.narrative_stack.common import UsGaapStore
from db import DbUsGaap
from simd_r_drive_ws_client import DataStoreWsClient


# Get the directory containing the script (do not change)
script_dir = os.path.dirname(os.path.abspath(__file__))


def test_ingestion_and_lookup():
    # Create test CSVs in temp dir
    csv_dir = os.path.join(script_dir, "assets/truncated_csvs")

    # Create in-memory or stub DB object
    db = DbUsGaap()

    # Connect to store
    host=os.getenv("SIMD_R_DRIVE_SERVER_HOST")
    port=os.getenv("SIMD_R_DRIVE_SERVER_PORT")
    data_store = DataStoreWsClient(f"{host}:{port}")

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

    print("Fetching cached data...")
    cached_data = us_gaap_store.batch_lookup_by_indices(list(range(triplet_count)))

    print(f"Cached data length: {len(cached_data)}")
    assert len(cached_data) == triplet_count

    # Inverse scaling
    has_unscaled_value_check = False
    for i in range(0, triplet_count):
        data = cached_data[i]

        # Sanity check to ensure the scaler is actually working
        if data["unscaled_value"] != 0:
            assert data["unscaled_value"] != data["scaled_value"]

        # For 64-bit internal values
        transformed = data["scaler"].transform([[data["unscaled_value"]]])[0][0]
        assert (
            transformed == data["scaled_value"]
        ), f"Expected {data['scaled_value']}, but got {transformed}"

        # For 32-bit only
        # transformed = np.float32(
        #     data["scaler"].transform([[data["unscaled_value"]]])[0][0]
        # )
        # expected = np.float32(data["scaled_value"])
        # assert np.isclose(
        #     transformed, expected, rtol=1e-6, atol=1e-7
        # ), f"Expected {expected}, but got {transformed}"

        has_unscaled_value_check = True

    assert has_unscaled_value_check
