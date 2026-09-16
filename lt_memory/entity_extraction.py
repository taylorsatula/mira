"""
Entity extraction for memory linking.

Extracts named entities from memory text using spaCy NER with optimized
configuration (parser/lemmatizer disabled). Performs dynamic normalization
and fuzzy clustering without hardcoded entity lists.
"""
import logging
import threading
from typing import List, Dict, Set, Optional

from lt_memory.models import NamedEntity
from collections import defaultdict

import spacy
from rapidfuzz import fuzz

from config import config

logger = logging.getLogger(__name__)

# Process-wide spaCy model cache. The Language object is a user-agnostic,
# read-only resource; loading en_core_web_lg deserializes hundreds of MB from
# disk (seconds). EntityExtractor is constructed per tool invocation (the tool
# repository mints fresh tool instances per call), so without this cache every
# memory search would reload the model. The lock guards first load against
# concurrent tool threads and serializes NER inference, which spaCy does not
# document as thread-safe.
_NLP_MODEL = None
_NLP_LOCK = threading.Lock()


def _get_nlp_model():
    """Load en_core_web_lg once per process."""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        with _NLP_LOCK:
            if _NLP_MODEL is None:
                try:
                    _NLP_MODEL = spacy.load(
                        "en_core_web_lg",
                        disable=["parser", "lemmatizer", "textcat"]
                    )
                    logger.info("EntityExtractor initialized with en_core_web_lg (optimized)")
                except OSError:
                    logger.error(
                        "en_core_web_lg not found. Install with: "
                        "python -m spacy download en_core_web_lg"
                    )
                    raise
    return _NLP_MODEL


class EntityExtractor:
    """
    Fast entity extraction for memory linking.

    Uses en_core_web_lg with parser and lemmatizer disabled for efficiency.
    Dynamically normalizes entities using fuzzy clustering to handle
    variations (PostgreSQL/Postgres/postgres → normalized form).
    """

    # Entity types to extract (high-value for memory linking)
    ENTITY_TYPES = {
        "PERSON", "ORG", "GPE", "PRODUCT", "EVENT",
        "WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "FAC"
    }

    # Common verbs that indicate spaCy misclassified a verb phrase as an entity
    # e.g., "Built MIRA", "Uses Square", "Holds AI" are not valid entities
    VERB_PREFIXES = {
        "built", "uses", "holds", "wants", "needs", "likes", "loves",
        "hates", "makes", "creates", "develops", "works", "runs",
        "manages", "leads", "owns", "maintains", "prefers", "enjoys"
    }

    def __init__(self):
        """Initialize entity extractor with the shared spaCy model."""
        self.nlp = _get_nlp_model()

    def extract_entities(self, text: str) -> Set[str]:
        """
        Extract normalized entities from text.

        Args:
            text: Memory text to extract entities from

        Returns:
            Set of normalized entity names

        Raises:
            Exception: If spaCy NLP processing fails
        """
        if not text or len(text) < 10:
            return set()

        with _NLP_LOCK:
            doc = self.nlp(text)
        entities = set()

        for ent in doc.ents:
            # Filter by entity type
            if ent.label_ not in self.ENTITY_TYPES:
                continue

            # Skip verb phrases (spaCy misclassification)
            if self._is_verb_phrase(ent.text):
                continue

            # Normalize entity text
            normalized = self._normalize_entity(ent.text)
            if normalized:
                entities.add(normalized)

        return entities

    def extract_entities_with_types(self, text: str) -> List[NamedEntity]:
        """
        Extract normalized entities with their types from text.

        Args:
            text: Memory text to extract entities from

        Returns:
            List of NamedEntity dicts with name and entity_type

        Raises:
            Exception: If spaCy NLP processing fails
        """
        if not text or len(text) < 10:
            return []

        with _NLP_LOCK:
            doc = self.nlp(text)
        entities: List[NamedEntity] = []

        for ent in doc.ents:
            # Filter by entity type
            if ent.label_ not in self.ENTITY_TYPES:
                continue

            # Skip verb phrases (spaCy misclassification)
            if self._is_verb_phrase(ent.text):
                continue

            # Normalize entity text
            normalized = self._normalize_entity(ent.text)
            if normalized:
                entities.append(NamedEntity(name=normalized, entity_type=ent.label_))

        return entities

    def extract_entities_batch(self, texts: List[str]) -> List[Set[str]]:
        """
        Extract entities from multiple texts in batch.

        Args:
            texts: List of memory texts

        Returns:
            List of entity sets (parallel to input)

        Raises:
            Exception: If batch processing fails
        """
        if not texts:
            return []

        # Process in batch under the shared-model lock: nlp.pipe mutates
        # internal pipeline state during batching and must not interleave with
        # other inference on the same Language object.
        results = []
        with _NLP_LOCK:
            docs = list(self.nlp.pipe(texts, batch_size=config.lt_memory.ner_batch_size))
        for doc in docs:
            entities = set()
            for ent in doc.ents:
                if ent.label_ in self.ENTITY_TYPES:
                    # Skip verb phrases (spaCy misclassification)
                    if self._is_verb_phrase(ent.text):
                        continue
                    normalized = self._normalize_entity(ent.text)
                    if normalized:
                        entities.add(normalized)
            results.append(entities)

        return results

    def _is_verb_phrase(self, text: str) -> bool:
        """
        Check if entity starts with a common verb (spaCy misclassification).

        Examples: "Built MIRA", "Uses Square", "Holds AI" are verb phrases,
        not valid entity names.

        Args:
            text: Entity text to check

        Returns:
            True if entity appears to be a verb phrase
        """
        if not text:
            return False
        first_word = text.split()[0].lower()
        return first_word in self.VERB_PREFIXES

    def _normalize_entity(self, entity_text: str) -> Optional[str]:
        """
        Normalize entity text for consistent linking.

        Simple normalization:
        - Strip whitespace
        - Collapse multiple spaces
        - Preserve original casing (important for proper nouns)

        Args:
            entity_text: Raw entity text from spaCy

        Returns:
            Normalized entity name or None if invalid
        """
        if not entity_text:
            return None

        # Clean: strip and collapse whitespace
        cleaned = " ".join(entity_text.strip().split())

        # Filter very short or invalid entities
        if len(cleaned) < 2:
            return None

        return cleaned

    def cluster_similar_entities(
        self,
        entities: Set[str],
        similarity_threshold: float = 0.85
    ) -> Dict[str, str]:
        """
        Cluster similar entity variations to canonical forms.

        Uses fuzzy matching to map variations:
        - "PostgreSQL" ← "postgres", "Postgres", "PostgreSQL"
        - "OpenAI" ← "openai", "Open AI", "OpenAI"

        Args:
            entities: Set of entity strings to cluster
            similarity_threshold: Fuzzy match threshold (0.0-1.0)

        Returns:
            Dictionary mapping variant → canonical form
        """
        if not entities:
            return {}

        entity_list = sorted(entities, key=len, reverse=True)  # Longer = more canonical
        canonical_map = {}
        used_canonical = set()

        for entity in entity_list:
            # Skip if already mapped as a variant
            if entity in canonical_map:
                continue

            # Find if this matches an existing canonical form
            best_match = None
            best_score = 0

            for canonical in used_canonical:
                score = fuzz.ratio(entity.lower(), canonical.lower()) / 100.0
                if score >= similarity_threshold and score > best_score:
                    best_match = canonical
                    best_score = score

            if best_match:
                # Map to existing canonical
                canonical_map[entity] = best_match
            else:
                # This becomes a new canonical form
                canonical_map[entity] = entity
                used_canonical.add(entity)

        return canonical_map

    def find_shared_entities(
        self,
        entity_sets: List[Set[str]]
    ) -> Dict[str, List[int]]:
        """
        Find entities shared across multiple memory entity sets.

        Performs fuzzy clustering across all entities first to handle
        variations, then identifies which memories share each canonical entity.

        Args:
            entity_sets: List of entity sets from different memories

        Returns:
            Dictionary mapping canonical entity → list of memory indices containing it
        """
        # Collect all unique entities
        all_entities = set()
        for entities in entity_sets:
            all_entities.update(entities)

        # Cluster similar entities to canonical forms
        entity_to_canonical = self.cluster_similar_entities(all_entities)

        # Map canonical entities to memory indices
        canonical_to_indices = defaultdict(list)

        for idx, entities in enumerate(entity_sets):
            canonical_entities = {entity_to_canonical[e] for e in entities}
            for canonical in canonical_entities:
                canonical_to_indices[canonical].append(idx)

        # Filter to only entities appearing in 2+ memories
        shared = {
            entity: indices
            for entity, indices in canonical_to_indices.items()
            if len(indices) >= 2
        }

        return shared

    def cleanup(self) -> None:
        """
        Release spaCy model resources.

        No-op: spaCy model managed by factory lifecycle.
        Nulling reference breaks in-flight scheduler jobs.
        """
        logger.debug("EntityExtractor cleanup completed (no-op)")
