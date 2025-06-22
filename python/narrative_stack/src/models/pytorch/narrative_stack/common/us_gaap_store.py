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
        full_batch = []

        for pair, i_cells in tqdm(
            concept_unit_pairs_i_cells.items(), desc="Scaling per concept/unit"
        ):
            i_bytes_list = [i.to_bytes(4, "little", signed=False) for i in i_cells]
            keys = [
                UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
                for i_bytes in i_bytes_list
            ]

            # Use read and decode float directly
            values = []
            for key in keys:
                # MODIFIED: Changed read_entry().as_memoryview() to read()
                raw_bytes = self.data_store.read(key)
                if raw_bytes is None:
                    raise KeyError(f"Missing unscaled value for key {key}")
                values.append(_decode_float_from_bytes(raw_bytes))

            vals_np = np.array(values).reshape(-1, 1)

            n_q = min(len(values), 1000)
            if n_q < 2 and len(values) >= 2:
                n_q = 2

            # --- ALWAYS USE A SCALER ---

            if len(values) < 2:
                # Fallback to a StandardScaler for numeric stability
                scaler = StandardScaler()
                scaled_vals = scaler.fit_transform(vals_np).flatten()

            else:
                scaler = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=n_q,
                    subsample=len(values),
                    random_state=42,
                )
                scaled_vals = scaler.fit_transform(vals_np).flatten()

            # Store the fitted scaler (encoded with joblib helper)
            # This line will now always have 'scaler' defined
            scaler_bytes_encoded = _encode_joblib_object_to_bytes(scaler)
            pair_id = self._get_pair_id(pair)
            self.data_store.write(
                SCALER_NAMESPACE.namespace(pair_id.to_bytes(4, "little")),
                scaler_bytes_encoded,
            )

            assert len(scaled_vals) == len(i_cells)

            # TODO: Uncomment and replace `full_batch` if memory is an issue here
            # Store scaled values (encoded as float64 bytes)
            # self.data_store.batch_write(
            #     [
            #         (
            #             SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(
            #                 i.to_bytes(4, "little", signed=False)
            #             ),
            #             _encode_float_to_raw_bytes(val),
            #         )
            #         for i, val in zip(i_cells, scaled_vals)
            #     ]
            # )

            for i, val in zip(i_cells, scaled_vals):
                full_batch.append(
                    (
                        SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(
                            i.to_bytes(4, "little", signed=False)
                        ),
                        _encode_float_to_raw_bytes(val),
                    )
                )

        self.data_store.batch_write(full_batch)

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

    # TODO: Implement `batch_lookup_by_indices`

    def lookup_by_index(self, i_cell: int) -> dict:
        i_bytes = i_cell.to_bytes(4, "little", signed=False)

        # Load concept_unit_id from cell meta (raw int)
        meta_key = CELL_META_NAMESPACE.namespace(i_bytes)
        concept_unit_id_bytes = self.data_store.read(meta_key)
        if concept_unit_id_bytes is None:
            raise KeyError(f"Missing concept_unit_id for i_cell {i_cell}")
        pair_id = int.from_bytes(concept_unit_id_bytes, "little", signed=False)

        # Load (concept, uom) using length-prefixed strings
        pair_key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(
            pair_id.to_bytes(4, "little", signed=False)
        )
        pair_bytes = self.data_store.read(pair_key)
        if pair_bytes is None:
            raise KeyError(f"Missing (concept, uom) for concept_unit_id {pair_id}")
        concept, offset = _decode_string_from_bytes(pair_bytes, 0)
        uom, _ = _decode_string_from_bytes(pair_bytes, offset)

        # Load unscaled value (raw float64)
        unscaled_key = UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
        unscaled_bytes = self.data_store.read(unscaled_key)
        if unscaled_bytes is None:
            raise KeyError(f"Missing unscaled value for i_cell {i_cell}")
        unscaled_value = _decode_float_from_bytes(unscaled_bytes)

        # Load scaled value (optional, raw float64)
        scaled_key = SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
        scaled_bytes = self.data_store.read(scaled_key)
        scaled_value = None
        if scaled_bytes is not None:
            scaled_value = _decode_float_from_bytes(scaled_bytes)

        # Load PCA-reduced embedding (raw float64 numpy array)
        embedding_key = PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(
            pair_id.to_bytes(4, "little", signed=False)
        )
        embedding_bytes = self.data_store.read(embedding_key)
        if embedding_bytes is None:
            raise KeyError(f"Missing embedding for concept_unit_id {pair_id}")
        embedding = _decode_numpy_array_from_bytes(embedding_bytes, dtype=np.float64)

        # Load scaler (cached, joblib)
        scaler_key = SCALER_NAMESPACE.namespace(
            pair_id.to_bytes(4, "little", signed=False)
        )

        # Check cache first
        if scaler_key in self._scaler_cache:
            # If in cache, retrieve it directly. No further loading needed.
            scaler = self._scaler_cache[scaler_key]
        else:
            # Cache miss: Attempt to load from store
            scaler_bytes = self.data_store.read(scaler_key)
            if scaler_bytes is not None:
                # Found in store: decode, assign to scaler, and cache it
                loaded_scaler = _decode_joblib_object_from_bytes(scaler_bytes)
                scaler = loaded_scaler
            else:
                # Not found in store: Assign None, and cache None
                scaler = None
            self._scaler_cache[scaler_key] = scaler

        # At this point, 'scaler' is guaranteed to be associated with a value
        return {
            "i_cell": i_cell,
            "pair_id": pair_id,
            "concept": concept,
            "uom": uom,
            "unscaled_value": unscaled_value,
            "scaled_value": scaled_value,
            "embedding": embedding,
            "scaler": scaler,
        }

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
