import logging
import joblib
from io import BytesIO
import torch
from sentence_transformers import SentenceTransformer
from simd_r_drive_ws_client import DataStoreWsClient, NamespaceHasher

from utils.csv import walk_us_gaap_csvs
from collections import defaultdict

from tqdm import tqdm
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Iterator, Optional, Any
from utils.pytorch import model_hash, get_device, seed_everything
from utils import generate_us_gaap_description
from models.pytorch.narrative_stack.stage1.preprocessing import (
    pca_compress_concept_unit_embeddings,
)

# Note: This is used here for the semantic modeling (BGE model)
seed_everything()

# --- NAMESPACES ---
TRIPLET_REVERSE_INDEX_NAMESPACE = NamespaceHasher(b"triplet-reverse-index")
UNSCALED_SEQUENTIAL_CELL_NAMESPACE = NamespaceHasher(b"unscaled-sequential-cell")
SCALED_SEQUENTIAL_CELL_NAMESPACE = NamespaceHasher(b"scaled-sequential-cell")
CELL_META_NAMESPACE = NamespaceHasher(b"cell-meta")
CONCEPT_UNIT_PAIR_NAMESPACE = NamespaceHasher(b"concept-unit-pair")
SCALER_NAMESPACE = NamespaceHasher(b"scaler")
PCA_MODEL_NAMESPACE = NamespaceHasher(b"pca-model")
PCA_REDUCED_EMBEDDING_NAMESPACE = NamespaceHasher(b"pca-reduced-embedding")

# --- GLOBAL CONSTANTS FOR ENCODING/DECODING ---
LEN_PREFIX_BYTES = 2  # Use 2 bytes for string length prefixes (up to 65535 bytes)

# --- HELPER FUNCTIONS FOR ENCODING/DECODING (UPDATED TO USE BYTES) ---


def _encode_string_to_bytes(s: str) -> bytes:
    """Encodes a string to UTF-8 bytes with a length prefix."""
    s_bytes = s.encode("utf-8")
    s_len = len(s_bytes)
    if s_len > (2 ** (8 * LEN_PREFIX_BYTES) - 1):  # Check if length fits in prefix
        raise ValueError(
            f"String length {s_len} exceeds max for {LEN_PREFIX_BYTES} bytes prefix."
        )
    return s_len.to_bytes(LEN_PREFIX_BYTES, "little") + s_bytes


def _decode_string_from_bytes(data: bytes, offset: int) -> Tuple[str, int]:
    """Decodes a length-prefixed string from a bytes object."""
    s_len = int.from_bytes(data[offset : offset + LEN_PREFIX_BYTES], "little")
    offset += LEN_PREFIX_BYTES
    s = data[offset : offset + s_len].decode("utf-8")
    offset += s_len
    return s, offset


def _encode_float_to_raw_bytes(f: float) -> bytes:
    """Encodes a single float to raw float64 bytes."""
    return np.array([f], dtype=np.float64).tobytes()


def _decode_float_from_bytes(data: bytes) -> float:
    """Decodes a single float from raw float64 bytes."""
    return np.frombuffer(data, dtype=np.float64)[0]


def _encode_numpy_array_to_raw_bytes(arr: np.ndarray) -> bytes:
    """Encodes a numpy array (assumed float64) to raw bytes."""
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    return arr.tobytes()


def _decode_numpy_array_from_bytes(
    data: bytes, dtype: np.dtype, shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Decodes a numpy array from a bytes object."""
    arr = np.frombuffer(data, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def _encode_joblib_object_to_bytes(obj: Any) -> bytes:
    """Encodes a joblib-compatible object to bytes."""
    buffer = BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def _decode_joblib_object_from_bytes(data: bytes) -> Any:
    """Decodes a joblib-compatible object from a bytes object."""
    return joblib.load(BytesIO(data))


# --- Pydantic Model (No Change) ---
class ConceptUnitPair(BaseModel):
    concept: str
    uom: str

    class Config:
        frozen = True


# --- UsGaapStore Class ---
class UsGaapStore:
    def __init__(self, data_store: DataStoreWsClient):
        self.data_store = data_store
        # _pair_to_id_cache is only for ingestion, will be built during ingestion
        self._pair_to_id_cache: dict[ConceptUnitPair, int] = {}
        self._scaler_cache: dict[bytes, Any] = {}  # For runtime lookup

    # --- INGESTION METHODS ---
    # Note: `db_us_gaap` is assumed to be an external dependency or unused in this context
    def ingest_us_gaap_csvs(
        self, csv_data_dir: str, db_us_gaap: Any
    ):  # Changed type hint
        gen = walk_us_gaap_csvs(
            csv_data_dir, db_us_gaap, "row", tqdm_desc="Migrating CSV files..."
        )

        concept_unit_pairs_i_cells: dict[ConceptUnitPair, list[int]] = defaultdict(list)
        pair_to_id: dict[ConceptUnitPair, int] = {}
        concept_unit_entries: list[tuple[bytes, bytes]] = []

        i_cell = -1
        next_pair_id = 0

        try:
            while True:
                row = next(gen)
                batch = []

                for cell in row.entries:
                    i_cell += 1

                    pair = ConceptUnitPair(concept=cell.concept, uom=cell.uom)
                    i_bytes = i_cell.to_bytes(4, "little", signed=False)

                    if pair not in pair_to_id:
                        pair_to_id[pair] = next_pair_id
                        pair_id_bytes = next_pair_id.to_bytes(4, "little", signed=False)
                        pair_key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(pair_id_bytes)

                        # Encode concept and uom as length-prefixed strings
                        pair_val_encoded = _encode_string_to_bytes(
                            pair.concept
                        ) + _encode_string_to_bytes(pair.uom)
                        concept_unit_entries.append((pair_key, pair_val_encoded))
                        next_pair_id += 1

                    pair_id = pair_to_id[pair]
                    pair_id_bytes = pair_id.to_bytes(4, "little", signed=False)

                    concept_unit_pairs_i_cells[pair].append(i_cell)

                    # Store raw unscaled value (as float64 bytes)
                    unscaled_value_encoded = _encode_float_to_raw_bytes(cell.value)
                    unscaled_key = UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
                    batch.append((unscaled_key, unscaled_value_encoded))

                    # Store reverse triplet -> i_cell mapping (Custom Binary Triplet Key)
                    # Key is now: len_concept | concept_bytes | len_uom | uom_bytes | unscaled_value_float64_bytes
                    triplet_key_bytes = (
                        _encode_string_to_bytes(cell.concept)
                        + _encode_string_to_bytes(cell.uom)
                        + _encode_float_to_raw_bytes(cell.value)
                    )
                    triplet_key = TRIPLET_REVERSE_INDEX_NAMESPACE.namespace(
                        triplet_key_bytes
                    )

                    batch.append(
                        (triplet_key, i_bytes)
                    )  # i_bytes is already raw int bytes

                    # Store cell meta (i_cell -> concept_unit_id)
                    cell_meta_key = CELL_META_NAMESPACE.namespace(i_bytes)
                    batch.append((cell_meta_key, pair_id_bytes))

                self.data_store.batch_write(batch)

        except StopIteration as stop:
            summary = stop.value
            logging.info(summary)

        self._pair_to_id_cache = pair_to_id  # Cache for _get_pair_id during ingestion

        total_triplets = i_cell + 1
        self.data_store.write(
            b"__triplet_count__",
            total_triplets.to_bytes(4, byteorder="little", signed=False),
        )
        logging.info(f"Total triplets: {total_triplets}")

        self.data_store.batch_write(concept_unit_entries)

        total_pairs = len(concept_unit_pairs_i_cells)
        self.data_store.write(
            b"__pair_count__", total_pairs.to_bytes(4, "little", signed=False)
        )
        logging.info(f"Total concept/unit pairs: {total_pairs}")

        self._scale_values(concept_unit_pairs_i_cells)

    def _scale_values(self, concept_unit_pairs_i_cells):
        """
        Scales values for each concept/unit pair. It reads all unscaled values
        for a pair in performance-minded chunks, fits a single scaler to all
        of them, and then writes back the scaled values in batches.
        """
        READ_BATCH_SIZE = 1024

        full_write_batch = []

        for pair, i_cells in tqdm(
            concept_unit_pairs_i_cells.items(), desc="Scaling per concept/unit"
        ):
            # --- Step 1: Fetch all unscaled values for the current pair in chunks ---
            all_values_for_pair = []
            keys_for_pair = [
                UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(
                    i.to_bytes(4, "little", signed=False)
                )
                for i in i_cells
            ]

            for i in range(0, len(keys_for_pair), READ_BATCH_SIZE):
                chunk_keys = keys_for_pair[i : i + READ_BATCH_SIZE]
                raw_bytes_list = self.data_store.batch_read(chunk_keys)

                for raw_bytes, key in zip(raw_bytes_list, chunk_keys):
                    if raw_bytes is None:
                        raise KeyError(f"Missing unscaled value for key {key!r}")
                    all_values_for_pair.append(_decode_float_from_bytes(raw_bytes))

            # --- Step 2: Fit a single scaler to all values for this pair ---
            vals_np = np.array(all_values_for_pair).reshape(-1, 1)

            n_q = min(len(all_values_for_pair), 1000)
            if n_q < 2 and len(all_values_for_pair) >= 2:
                n_q = 2

            if len(all_values_for_pair) < 2:
                # Fallback to a StandardScaler for numeric stability
                scaler = StandardScaler()
            else:
                scaler = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=n_q,
                    subsample=len(all_values_for_pair),  # Use all values for fitting
                    random_state=42,
                )

            # Fit and transform all values at once
            scaled_vals = scaler.fit_transform(vals_np).flatten()

            # --- Step 3: Store the single fitted scaler for this pair ---
            scaler_bytes_encoded = _encode_joblib_object_to_bytes(scaler)
            pair_id = self._get_pair_id(pair)
            self.data_store.write(
                SCALER_NAMESPACE.namespace(pair_id.to_bytes(4, "little")),
                scaler_bytes_encoded,
            )

            assert len(scaled_vals) == len(i_cells)

            for i_cell, scaled_val in zip(i_cells, scaled_vals):
                scaled_val_key = SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(
                    i_cell.to_bytes(4, "little", signed=False)
                )
                scaled_val_bytes = _encode_float_to_raw_bytes(scaled_val)
                full_write_batch.append((scaled_val_key, scaled_val_bytes))

        # --- Step 4: Write back all scaled values in batches ---
        self.data_store.batch_write(full_write_batch)

    def _get_pair_id(self, pair: ConceptUnitPair) -> int:
        """
        Retrieves pair_id during ingestion.
        For runtime/lookup, should use get_pair_id which reads from store.
        """
        if pair in self._pair_to_id_cache:
            return self._pair_to_id_cache[pair]
        raise KeyError(f"Concept/unit pair not found in cache: {pair}")

    # --- EMBEDDING GENERATION/LOADING METHODS ---

    def generate_pca_embeddings(self):
        pairs = []
        embeddings = []

        for pair_id, pair, embedding in tqdm(
            self._generate_concept_unit_embeddings(get_device()),
            desc="Generating Semantic Embeddings",
        ):
            pairs.append((pair_id, pair))
            embeddings.append(embedding)  # Embedding is already np.ndarray of float64

        embedding_matrix = np.stack(embeddings, axis=0)
        logging.info(f"Embedding matrix shape: {embedding_matrix.shape}")

        pca_compressed_concept_unit_embeddings, pca = (
            pca_compress_concept_unit_embeddings(
                embedding_matrix,
                n_components=243,
                pca=None,
                stable=True,
            )
        )

        assert len(pairs) == len(pca_compressed_concept_unit_embeddings)

        # Store PCA-reduced embeddings (encoded as raw float64 numpy bytes)
        pca_embedding_entries = [
            (
                PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(
                    pair_id.to_bytes(4, "little", signed=False)
                ),
                _encode_numpy_array_to_raw_bytes(vec),  # Encode numpy array directly
            )
            for (pair_id, _), vec in zip(pairs, pca_compressed_concept_unit_embeddings)
        ]

        self.data_store.batch_write(pca_embedding_entries)

        logging.info(
            f"Wrote {len(pca_embedding_entries)} PCA-compressed embeddings to store."
        )

        # Store PCA model (encoded with joblib helper)
        pca_model_bytes_encoded = _encode_joblib_object_to_bytes(pca)
        self.data_store.write(
            PCA_MODEL_NAMESPACE.namespace(b"model"), pca_model_bytes_encoded
        )
        logging.info("Stored PCA model in store.")

    def load_pca_model(self) -> Optional[PCA]:
        # MODIFIED: Changed read_entry().as_memoryview() to read()
        pca_model_bytes = self.data_store.read(PCA_MODEL_NAMESPACE.namespace(b"model"))
        if pca_model_bytes is None:
            return None
        return _decode_joblib_object_from_bytes(pca_model_bytes)

    def _generate_concept_unit_embeddings(
        self,
        device: torch.device,
        batch_size: int = 64,
    ) -> Iterator[Tuple[int, ConceptUnitPair, np.ndarray]]:
        # This part generates np.ndarray, which then gets passed to generate_pca_embeddings
        # The SentenceTransformer part is external to the DataStore read/write process.
        def _embed_batch(pair_ids, pairs, texts, model, device):
            tokens = model.tokenize(texts)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                output = model.forward(tokens)
                embeddings = output["sentence_embedding"].cpu().numpy()
            for pair_id, pair, embedding in zip(pair_ids, pairs, embeddings):
                yield pair_id, pair, embedding

        pairs_iter = self.iterate_concept_unit_pairs()

        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        model.eval()  # IMPORTANT!
        model.to(device)

        logging.info(f"Embedding model hash: {model_hash(model)}")

        buffer_ids = []
        buffer_pairs = []
        buffer_texts = []

        for pair_id, pair in pairs_iter:
            text = (
                f"{generate_us_gaap_description(pair.concept)} measured in {pair.uom}"
            )
            buffer_ids.append(pair_id)
            buffer_pairs.append(pair)
            buffer_texts.append(text)

            if len(buffer_pairs) == batch_size:
                yield from _embed_batch(
                    buffer_ids, buffer_pairs, buffer_texts, model, device
                )
                buffer_ids.clear()
                buffer_pairs.clear()
                buffer_texts.clear()

        if buffer_pairs:
            yield from _embed_batch(
                buffer_ids, buffer_pairs, buffer_texts, model, device
            )

    # TODO: Use batch reads
    def get_embedding_matrix(self) -> Tuple[np.ndarray, list]:
        embedding_matrix = []
        pairs = []

        # MODIFIED: Changed read_entry().as_memoryview() to read()
        raw_bytes = self.data_store.read(b"__pair_count__")
        if raw_bytes is None:
            raise KeyError("Missing __pair_count__ key in store")
        total_pairs = int.from_bytes(raw_bytes, "little", signed=False)

        for pair_id in range(total_pairs):
            pair_id_bytes = pair_id.to_bytes(4, "little", signed=False)
            pair_key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(pair_id_bytes)
            # MODIFIED: Changed read_entry().as_memoryview() to read()
            pair_bytes = self.data_store.read(pair_key)
            if pair_bytes is None:
                raise KeyError(f"Missing concept/unit for pair_id {pair_id}")

            concept, offset = _decode_string_from_bytes(pair_bytes, 0)
            uom, _ = _decode_string_from_bytes(pair_bytes, offset)

            # Load PCA-reduced embedding (direct numpy)
            embedding_key = PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(pair_id_bytes)
            # MODIFIED: Changed read_entry().as_memoryview() to read()
            embedding_bytes = self.data_store.read(embedding_key)
            if embedding_bytes is None:
                raise KeyError(f"Missing embedding for pair_id {pair_id}")
            embedding = _decode_numpy_array_from_bytes(
                embedding_bytes, dtype=np.float64
            )

            pairs.append((pair_id, ConceptUnitPair(concept=concept, uom=uom)))
            embedding_matrix.append(embedding)

        embedding_matrix_np = np.stack(embedding_matrix, axis=0)
        return embedding_matrix_np, pairs

    # --- CORE LOOKUP METHODS (OPTIMIZED) ---

    def get_triplet_count(self) -> int:
        # MODIFIED: Changed read_entry().as_memoryview() to read()
        raw_bytes = self.data_store.read(b"__triplet_count__")
        if raw_bytes is None:
            raise KeyError("Triplet count key not found")
        return int.from_bytes(raw_bytes, "little", signed=False)

    def get_pair_count(self) -> int:
        # MODIFIED: Changed read_entry().as_memoryview() to read()
        raw_bytes = self.data_store.read(b"__pair_count__")
        if raw_bytes is None:
            raise KeyError("Pair count key not found")
        return int.from_bytes(raw_bytes, "little", signed=False)

    def iterate_concept_unit_pairs(self) -> Iterator[Tuple[int, ConceptUnitPair]]:
        total_pairs = self.get_pair_count()
        for pair_id in range(total_pairs):
            key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(
                pair_id.to_bytes(4, "little", signed=False)
            )
            # MODIFIED: Changed read_entry().as_memoryview() to read()
            pair_bytes = self.data_store.read(key)
            if pair_bytes is None:
                raise KeyError(f"Missing concept/unit for pair_id={pair_id}")

            pair_mv = pair_bytes
            concept, offset = _decode_string_from_bytes(pair_mv, 0)
            uom, _ = _decode_string_from_bytes(pair_mv, offset)

            yield (pair_id, ConceptUnitPair(concept=concept, uom=uom))

    def lookup_by_index(self, i_cell: int) -> dict:
        batch_results = self.batch_lookup_by_indices([i_cell])
        return batch_results[0]

    def batch_lookup_by_indices(self, cell_indices: list[int]) -> list[dict]:
        # TODO: Don't name these "stage1", etc. since various stages of the model are also named this way

        # Stage 1: Fetch meta and cell-specific values
        stage1_requests = []
        for i_cell in cell_indices:
            i_bytes = i_cell.to_bytes(4, "little")
            stage1_requests.append(
                {
                    "meta": CELL_META_NAMESPACE.namespace(i_bytes),
                    "unscaled": UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes),
                    "scaled": SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes),
                }
            )

        stage1_results = self.data_store.batch_read_structured(stage1_requests)

        # Stage 2: Process stage 1, gather unique pair_ids and build stage 2 requests
        pair_id_map = {}  # Maps i_cell -> pair_id
        stage2_requests = {}  # Maps pair_id -> request dict to avoid duplicate fetches
        for i_cell, result in zip(cell_indices, stage1_results):
            meta_bytes = result["meta"]
            if meta_bytes is None:
                raise KeyError(f"Missing meta for i_cell {i_cell}")
            pair_id = int.from_bytes(meta_bytes, "little")
            pair_id_map[i_cell] = pair_id

            if pair_id not in stage2_requests:
                pair_id_bytes = pair_id.to_bytes(4, "little")
                request = {
                    "pair_info": CONCEPT_UNIT_PAIR_NAMESPACE.namespace(pair_id_bytes),
                    "embedding": PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(
                        pair_id_bytes
                    ),
                }
                # Only fetch scaler if it's not in the cache
                if SCALER_NAMESPACE.namespace(pair_id_bytes) not in self._scaler_cache:
                    request["scaler"] = SCALER_NAMESPACE.namespace(pair_id_bytes)
                stage2_requests[pair_id] = request

        # Stage 3: Fetch pair-dependent data
        unique_pair_ids = list(stage2_requests.keys())
        unique_requests = [stage2_requests[pid] for pid in unique_pair_ids]
        stage2_results_list = self.data_store.batch_read_structured(unique_requests)
        stage2_results_map = dict(zip(unique_pair_ids, stage2_results_list))

        # Stage 4: Consolidate results
        final_results = []
        for i_cell, s1_result in zip(cell_indices, stage1_results):
            pair_id = pair_id_map[i_cell]
            s2_result = stage2_results_map[pair_id]

            # Decode all data
            unscaled_bytes = s1_result["unscaled"]
            if unscaled_bytes is None:
                raise KeyError(f"Missing unscaled value for i_cell {i_cell}")

            pair_bytes = s2_result["pair_info"]
            if pair_bytes is None:
                raise KeyError(f"Missing pair info for pair_id {pair_id}")
            concept, offset = _decode_string_from_bytes(pair_bytes, 0)
            uom, _ = _decode_string_from_bytes(pair_bytes, offset)

            embedding_bytes = s2_result["embedding"]
            if embedding_bytes is None:
                raise KeyError(f"Missing embedding for pair_id {pair_id}")

            # Handle scaler caching
            scaler_key = SCALER_NAMESPACE.namespace(pair_id.to_bytes(4, "little"))
            if scaler_key in self._scaler_cache:
                scaler = self._scaler_cache[scaler_key]
            else:
                scaler_bytes = s2_result.get("scaler")
                scaler = (
                    _decode_joblib_object_from_bytes(scaler_bytes)
                    if scaler_bytes
                    else None
                )
                self._scaler_cache[scaler_key] = scaler

            final_results.append(
                {
                    "i_cell": i_cell,
                    "pair_id": pair_id,
                    "concept": concept,
                    "uom": uom,
                    "unscaled_value": _decode_float_from_bytes(unscaled_bytes),
                    "scaled_value": (
                        _decode_float_from_bytes(s1_result["scaled"])
                        if s1_result["scaled"]
                        else None
                    ),
                    "embedding": _decode_numpy_array_from_bytes(
                        embedding_bytes, dtype=np.float64
                    ),
                    "scaler": scaler,
                }
            )

        return final_results

    # TODO: Implement `batch_lookup_by_triplets`

    def lookup_by_triplet(self, concept: str, uom: str, unscaled_value: float) -> dict:
        """
        Given a (concept, uom, value) triplet, return its i_cell, unscaled value,
        and scaled value if available.
        NOTE: The key generation for this method must match the custom binary format
              used in ingest_us_gaap_csvs for TRIPLET_REVERSE_INDEX_NAMESPACE.
        """
        # Encode the triplet as used in reverse index (CUSTOM BINARY FORMAT)
        triplet_key_bytes = (
            _encode_string_to_bytes(concept)
            + _encode_string_to_bytes(uom)
            + _encode_float_to_raw_bytes(unscaled_value)
        )
        triplet_key = TRIPLET_REVERSE_INDEX_NAMESPACE.namespace(triplet_key_bytes)

        i_cell_bytes = self.data_store.read(triplet_key)
        if i_cell_bytes is None:
            raise KeyError(
                f"Triplet ({concept}, {uom}, {unscaled_value}) not found in reverse index"
            )

        i_cell = int.from_bytes(i_cell_bytes, "little", signed=False)

        return self.lookup_by_index(i_cell)
