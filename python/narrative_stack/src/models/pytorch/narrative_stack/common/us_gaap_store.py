import logging
import joblib
from io import BytesIO
import torch
from sentence_transformers import SentenceTransformer
from simd_r_drive import DataStore, NamespaceHasher
from db import DbUsGaap
from utils.csv import walk_us_gaap_csvs
from collections import defaultdict
import msgpack
from tqdm import tqdm
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from typing import Tuple, Iterator, Optional, Any
from utils.pytorch import model_hash, get_device, seed_everything
from utils import generate_us_gaap_description
from models.pytorch.narrative_stack.stage1.preprocessing import (
    pca_compress_concept_unit_embeddings,
)

# Note: This is used here for the semantic modeling (BGE model)
seed_everything()

# Namespaces for storing structured data
TRIPLET_REVERSE_INDEX_NAMESPACE = NamespaceHasher(b"triplet-reverse-index")
UNSCALED_SEQUENTIAL_CELL_NAMESPACE = NamespaceHasher(b"unscaled-sequential-cell")
SCALED_SEQUENTIAL_CELL_NAMESPACE = NamespaceHasher(b"scaled-sequential-cell")
CELL_META_NAMESPACE = NamespaceHasher(b"cell-meta")
CONCEPT_UNIT_PAIR_NAMESPACE = NamespaceHasher(b"concept-unit-pair")
# REVERSE_CONCEPT_UNIT_PAIR_NAMESPACE = NamespaceHasher(b"reverse-pair-map") # TODO: Implement
SCALER_NAMESPACE = NamespaceHasher(b"scaler")

PCA_MODEL_NAMESPACE = NamespaceHasher(b"pca-model")
PCA_REDUCED_EMBEDDING_NAMESPACE = NamespaceHasher(b"pca-reduced-embedding")


# Define immutable concept/unit pair model
class ConceptUnitPair(BaseModel):
    concept: str
    uom: str

    class Config:
        # Enables hashing
        frozen = True


class UsGaapStore:
    def __init__(self, data_store: DataStore):
        self.data_store = data_store

        # TODO: Remove
        # Only available during ingestion
        self._pair_to_id_cache: dict[ConceptUnitPair, int] = {}

    # Note: Most methods do not require the database, so it's used as an
    # argument here
    def ingest_us_gaap_csvs(self, csv_data_dir: str, db_us_gaap: DbUsGaap):
        # Initialize CSV stream generator
        gen = walk_us_gaap_csvs(
            csv_data_dir, db_us_gaap, "row", tqdm_desc="Migrating CSV files..."
        )

        # Track per (concept, uom) the list of i_cell indices that use it
        concept_unit_pairs_i_cells: dict[ConceptUnitPair, list[int]] = defaultdict(list)
        pair_to_id: dict[ConceptUnitPair, int] = {}
        concept_unit_entries: list[tuple[bytes, bytes]] = []

        # Global sequential index for each cell value
        i_cell = -1
        next_pair_id = 0

        # Stream and store data
        try:
            while True:
                row = next(gen)
                batch = []

                for cell in row.entries:
                    i_cell += 1

                    pair = ConceptUnitPair(concept=cell.concept, uom=cell.uom)
                    i_bytes = i_cell.to_bytes(4, "little", signed=False)

                    # Assign ID to concept/unit pair if not already done
                    if pair not in pair_to_id:
                        pair_to_id[pair] = next_pair_id
                        pair_id_bytes = next_pair_id.to_bytes(4, "little", signed=False)
                        pair_key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(pair_id_bytes)
                        pair_val = msgpack.packb((pair.concept, pair.uom))
                        concept_unit_entries.append((pair_key, pair_val))
                        next_pair_id += 1

                        # TODO: Implement
                        # reverse_key = REVERSE_CONCEPT_UNIT_PAIR_NAMESPACE.namespace(
                        #     msgpack.packb((pair.concept, pair.uom))
                        # )
                        # store.write(reverse_key, pair_id_bytes)

                    pair_id = pair_to_id[pair]
                    pair_id_bytes = pair_id.to_bytes(4, "little", signed=False)

                    # Track cell indices per (concept, uom)
                    concept_unit_pairs_i_cells[pair].append(i_cell)

                    # Store raw unscaled value
                    value_bytes = msgpack.packb(cell.value)
                    unscaled_key = UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
                    batch.append((unscaled_key, value_bytes))

                    # Store reverse triplet → i_cell mapping
                    triplet_bytes = msgpack.packb((cell.concept, cell.uom, cell.value))
                    triplet_key = TRIPLET_REVERSE_INDEX_NAMESPACE.namespace(
                        triplet_bytes
                    )
                    batch.append((triplet_key, i_bytes))

                    # Store cell meta (i_cell → concept_unit_id)
                    cell_meta_key = CELL_META_NAMESPACE.namespace(i_bytes)
                    batch.append((cell_meta_key, pair_id_bytes))

                # Write current batch of entries to store
                self.data_store.batch_write(batch)

                # TODO: Comment-out
                # Optional cutoff for debugging
                # if i_cell > 1000:
                #     break

        except StopIteration as stop:
            summary = stop.value
            logging.info(summary)

        # TODO: Remove
        self._pair_to_id_cache = pair_to_id

        total_triplets = i_cell + 1

        self.data_store.write(
            b"__triplet_count__",
            total_triplets.to_bytes(4, byteorder="little", signed=False),
        )

        logging.info(f"Total triplets: {total_triplets}")

        # Persist concept_unit_id → (concept, uom) mapping
        self.data_store.batch_write(concept_unit_entries)

        total_pairs = len(concept_unit_pairs_i_cells)

        self.data_store.write(
            b"__pair_count__", total_pairs.to_bytes(4, byteorder="little", signed=False)
        )

        # Show number of unique concept/unit pairs
        logging.info(f"Total concept/unit pairs: {total_pairs}")

        # # Show binary keys for each concept/unit pair
        # for pair in tqdm(
        #     concept_unit_pairs_i_cells, desc="Tracking concept/unit pairs"
        # ):
        #     logging(msgpack.packb((pair.concept, pair.uom)))

        self._scale_values(concept_unit_pairs_i_cells)

    def _scale_values(self, concept_unit_pairs_i_cells):
        # Scale all values per concept/unit group
        for pair, i_cells in tqdm(
            concept_unit_pairs_i_cells.items(), desc="Scaling per concept/unit"
        ):
            i_bytes_list = [i.to_bytes(4, "little", signed=False) for i in i_cells]
            keys = [
                UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
                for i_bytes in i_bytes_list
            ]

            values = [
                msgpack.unpackb(self.data_store.read(key), raw=True) for key in keys
            ]

            vals_np = np.array(values).reshape(-1, 1)

            # Clamp quantiles based on sample size
            n_q = min(len(values), 1000)
            if n_q < 2 and len(values) >= 2:
                n_q = 2

            if len(values) < 2:
                logging.warning(
                    "Only one value present for concept/unit pair. Scaling skipped. This is to avoid inconsistencies in scaling for singleton groups."
                )

                # TODO: Consider using global scaled mean instead of 0
                # For singleton values, set them to 0 (or another fixed value)
                scaled_vals = np.zeros_like(vals_np.flatten())  # Set to 0
            else:
                scaler = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=n_q,
                    subsample=len(values),
                    random_state=42,
                )

                scaled_vals = scaler.fit_transform(vals_np).flatten()

            # Store the fitted scaler for future use (serialized with joblib)
            scaler_bytes = BytesIO()
            joblib.dump(scaler, scaler_bytes)
            scaler_bytes.seek(0)
            pair_id = self._get_pair_id(pair)
            self.data_store.write(
                SCALER_NAMESPACE.namespace(pair_id.to_bytes(4, "little")),
                scaler_bytes.read(),
            )

            assert len(scaled_vals) == len(i_cells)

            self.data_store.batch_write(
                [
                    (
                        SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(
                            i.to_bytes(4, "little", signed=False)
                        ),
                        msgpack.packb(val),
                    )
                    for i, val in zip(i_cells, scaled_vals)
                ]
            )

    # TODO: Replace w/ a store read
    # Only available during ingestion
    def _get_pair_id(self, pair: ConceptUnitPair) -> int:
        if pair in self._pair_to_id_cache:
            return self._pair_to_id_cache[pair]
        raise KeyError(f"Concept/unit pair not found in cache: {pair}")

    # TODO: Use
    # def get_pair_id(self, pair: ConceptUnitPair) -> int:
    #     key = REVERSE_CONCEPT_UNIT_PAIR_NAMESPACE.namespace(
    #         msgpack.packb((pair.concept, pair.uom))
    #     )
    #     raw = self.data_store.read(key)
    #     if raw is None:
    #         raise KeyError(f"Concept/unit pair not found: {pair}")
    #     return int.from_bytes(raw, "little")

    def iterate_concept_unit_pairs(self) -> Iterator[Tuple[int, ConceptUnitPair]]:
        """
        Yields (pair_id, ConceptUnitPair) from the concept/unit pair namespace.
        """
        raw = self.data_store.read(b"__pair_count__")
        if raw is None:
            raise ValueError("Missing __pair_count__ key in store")

        total_pairs = int.from_bytes(raw, "little", signed=False)

        for pair_id in range(total_pairs):
            key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(
                pair_id.to_bytes(4, "little", signed=False)
            )
            val = self.data_store.read(key)
            if val is None:
                raise KeyError(f"Missing concept/unit for pair_id={pair_id}")
            concept, uom = msgpack.unpackb(val, raw=True)
            yield (
                pair_id,
                ConceptUnitPair(
                    concept=concept.decode("utf-8"), uom=uom.decode("utf-8")
                ),
            )

    # TODO: Enable generation from an existing PCA
    def generate_pca_embeddings(self):
        pairs = []
        embeddings = []

        for pair_id, pair, embedding in tqdm(
            self._generate_concept_unit_embeddings(get_device()),
            desc="Generating Semantic Embeddings",
        ):
            pairs.append((pair_id, pair))
            embeddings.append(embedding)

        # Convert to NumPy array of shape (N, D)
        embedding_matrix = np.stack(embeddings, axis=0)

        # Now ready to pass `embedding_matrix` to PCA
        logging.info(f"Embedding matrix shape: {embedding_matrix.shape}")

        # TODO: Reuse PCA if already existing (provide ability to pull from another store, etc.)

        pca_compressed_concept_unit_embeddings, pca = (
            pca_compress_concept_unit_embeddings(
                # TODO: Don't hardcode 243
                embedding_matrix,
                n_components=243,
                pca=None,
                stable=True,
            )
        )

        assert len(pairs) == len(pca_compressed_concept_unit_embeddings)

        # TODO: Save PCA-reduced embeddings in store
        pca_embedding_entries = [
            (
                PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(
                    pair_id.to_bytes(4, "little", signed=False)
                ),
                msgpack.packb(
                    vec.astype(np.float32).tolist()
                ),  # Convert numpy array to list
            )
            for (pair_id, _), vec in zip(pairs, pca_compressed_concept_unit_embeddings)
        ]

        self.data_store.batch_write(pca_embedding_entries)
        logging.info(
            f"Wrote {len(pca_embedding_entries)} PCA-compressed embeddings to store."
        )

        # Serialize PCA model into a byte stream
        pca_model_stream = BytesIO()
        joblib.dump(pca, pca_model_stream)
        pca_model_stream.seek(0)  # Move cursor to the beginning of the stream

        # Store PCA model in the DataStore
        self.data_store.write(
            PCA_MODEL_NAMESPACE.namespace(b"model"), pca_model_stream.read()
        )

        logging.info("Stored PCA model in store.")

    def load_pca_model(self) -> Optional[PCA]:
        """
        Retrieve the PCA model from the store and deserialize it.

        Returns:
            PCA or None: The loaded PCA model, or None if not found.

        Raises:
            KeyError: If the PCA model is not found in the store.
        """
        # Retrieve the PCA model from store
        pca_model_bytes = self.data_store.read(PCA_MODEL_NAMESPACE.namespace(b"model"))
        if pca_model_bytes is None:
            return None  # PCA model doesn't exist in the store

        # Deserialize the PCA model from the byte stream
        pca_model_stream = BytesIO(pca_model_bytes)
        return joblib.load(pca_model_stream)

    def _generate_concept_unit_embeddings(
        self,
        device: torch.device,
        batch_size: int = 64,
    ) -> Iterator[Tuple[int, ConceptUnitPair, np.ndarray]]:
        """
        Yields (pair_id, concept_unit_pair, embedding) for each input concept/unit pair.
        """

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
        model.eval()
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
        """
        Retrieve all embeddings from the store and return the embedding matrix.

        Returns:
            - A tuple with:
                1. A NumPy array of shape (N, D), where N is the number of pairs and D is the dimensionality of the embedding.
                2. A list of pairs (pair_id, ConceptUnitPair) for reference.

        Raises:
            KeyError if any required value is missing.
        """
        embedding_matrix = []
        pairs = []

        # Retrieve the total number of pairs
        raw = self.data_store.read(b"__pair_count__")
        if raw is None:
            raise KeyError("Missing __pair_count__ key in store")

        total_pairs = int.from_bytes(raw, "little", signed=False)

        for pair_id in range(total_pairs):
            # Load (concept, uom) for the pair
            pair_id_bytes = pair_id.to_bytes(4, "little", signed=False)
            pair_key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(pair_id_bytes)
            pair_bytes = self.data_store.read(pair_key)
            if pair_bytes is None:
                raise KeyError(f"Missing (concept, uom) for pair_id {pair_id}")
            concept, uom = msgpack.unpackb(pair_bytes, raw=False)

            # Load PCA-reduced embedding for the pair
            embedding_key = PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(pair_id_bytes)
            embedding_bytes = self.data_store.read(embedding_key)
            if embedding_bytes is None:
                raise KeyError(f"Missing embedding for pair_id {pair_id}")
            embedding = msgpack.unpackb(embedding_bytes, raw=True)

            # Add pair and embedding to lists
            pairs.append((pair_id, ConceptUnitPair(concept=concept, uom=uom)))
            embedding_matrix.append(embedding)

        # Convert the embedding matrix to a NumPy array (shape: N x D)
        embedding_matrix_np = np.stack(embedding_matrix, axis=0)

        return embedding_matrix_np, pairs

    def get_triplet_count(self) -> int:
        raw = self.data_store.read(b"__triplet_count__")
        if raw is None:
            raise KeyError("Triplet count key not found")
        return int.from_bytes(raw, "little", signed=False)

    def get_pair_count(self) -> int:
        raw = self.data_store.read(b"__pair_count__")
        if raw is None:
            raise KeyError("Pair count key not found")
        return int.from_bytes(raw, "little", signed=False)

    def lookup_by_index(self, i_cell: int) -> dict:
        i_bytes = i_cell.to_bytes(4, "little", signed=False)

        # Load concept_unit_id from cell meta
        meta_key = CELL_META_NAMESPACE.namespace(i_bytes)
        concept_unit_id_bytes = self.data_store.read(meta_key)
        if concept_unit_id_bytes is None:
            raise KeyError(f"Missing concept_unit_id for i_cell {i_cell}")

        pair_id = int.from_bytes(concept_unit_id_bytes, "little", signed=False)

        # Load (concept, uom) from concept_unit_id
        pair_key = CONCEPT_UNIT_PAIR_NAMESPACE.namespace(
            pair_id.to_bytes(4, "little", signed=False)
        )
        pair_bytes = self.data_store.read(pair_key)
        if pair_bytes is None:
            raise KeyError(f"Missing (concept, uom) for concept_unit_id {pair_id}")

        concept, uom = msgpack.unpackb(pair_bytes, raw=False)

        # Load unscaled value
        unscaled_key = UNSCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
        unscaled_bytes = self.data_store.read(unscaled_key)
        if unscaled_bytes is None:
            raise KeyError(f"Missing unscaled value for i_cell {i_cell}")
        unscaled_value = msgpack.unpackb(unscaled_bytes, raw=True)

        # Load scaled value (optional)
        scaled_key = SCALED_SEQUENTIAL_CELL_NAMESPACE.namespace(i_bytes)
        scaled_bytes = self.data_store.read(scaled_key)
        scaled_value = (
            msgpack.unpackb(scaled_bytes, raw=True)
            if scaled_bytes is not None
            else None
        )

        # Load PCA-reduced embedding (using concept_unit_id, not i_cell)
        embedding_key = PCA_REDUCED_EMBEDDING_NAMESPACE.namespace(
            pair_id.to_bytes(4, "little", signed=False)
        )
        embedding_bytes = self.data_store.read(embedding_key)
        if embedding_bytes is None:
            raise KeyError(f"Missing embedding for concept_unit_id {pair_id}")
        # embedding = msgpack.unpackb(embedding_bytes, raw=True)
        embedding = np.array(
            msgpack.unpackb(embedding_bytes, raw=True), dtype=np.float32
        )

        # Load scaler
        scaler_key = SCALER_NAMESPACE.namespace(
            pair_id.to_bytes(4, "little", signed=False)
        )
        scaler_bytes = self.data_store.read(scaler_key)
        scaler: Optional[Any] = None
        if scaler_bytes is not None:
            scaler = joblib.load(BytesIO(scaler_bytes))

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

    def lookup_by_triplet(self, concept: str, uom: str, unscaled_value: float) -> dict:
        """
        Given a (concept, uom, value) triplet, return its i_cell, unscaled value,
        and scaled value if available.

        Returns a dict with keys: i_cell, unscaled_value, scaled_value
        """
        # Encode the triplet as used in reverse index
        triplet_key_bytes = msgpack.packb((concept, uom, unscaled_value))
        triplet_key = TRIPLET_REVERSE_INDEX_NAMESPACE.namespace(triplet_key_bytes)

        # Lookup i_cell
        i_cell_bytes = self.data_store.read(triplet_key)
        if i_cell_bytes is None:
            raise KeyError(
                f"Triplet ({concept}, {uom}, {unscaled_value}) not found in reverse index"
            )

        i_cell = int.from_bytes(i_cell_bytes, "little", signed=False)

        return self.lookup_by_index(i_cell)
