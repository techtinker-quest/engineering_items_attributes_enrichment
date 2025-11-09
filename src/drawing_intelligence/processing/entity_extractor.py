"""
Entity extraction module for the Drawing Intelligence System.

This module provides multi-layer entity extraction from OCR text using:
1. Regex patterns for fast, precise extraction (~95% of entities)
2. spaCy EntityRuler for context-aware extraction (~4% of entities)
3. Optional LLM enhancement for difficult cases (~1% of entities, cost-controlled)

Performance Characteristics:
    - Regex: ~50ms per page (fastest, deterministic)
    - spaCy: ~200ms per page (context-aware, requires model loading)
    - LLM: ~2-5s per page (most accurate, highest cost)

Threading Safety:
    WARNING: This class is NOT thread-safe due to shared spaCy model state.
    Create separate instances per thread or implement external locking.

Typical usage example:
    >>> config = EntityConfig(
    ...     use_llm=False,
    ...     confidence_threshold=0.80,
    ...     normalize_units=True
    ... )
    >>> with EntityExtractor(config, llm_gateway=None) as extractor:
    ...     result = extractor.extract_entities(ocr_result)
    ...     print(f"Found {len(result.entities)} entities")
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from ..models.data_structures import (
    Entity,
    EntityExtractionResult,
    EntityType,
    ExtractionStats,
    OCRResult,
    TextBlock,
    TitleBlock,
)
from ..utils.file_utils import generate_unique_id
from ..utils.geometry_utils import BoundingBox
from ..utils.text_utils import (
    convert_unit,
    extract_dimension_value,
    extract_part_number_candidates,
    normalize_technical_symbols,
)

logger = logging.getLogger(__name__)


class LLMGatewayProtocol(Protocol):
    """Protocol defining the interface for LLM gateway implementations."""

    def extract_entities_llm(
        self, text: str, context: str, entity_types: List[str]
    ) -> List[Entity]:
        """Extract entities using LLM."""
        ...


@dataclass
class EntityConfig:
    """
    Configuration for entity extraction behavior.

    Attributes:
        use_regex: Enable regex-based extraction for fast pattern matching.
        use_spacy: Enable spaCy-based extraction for context-aware NER.
        use_llm: Enable LLM-based extraction for complex cases (increases cost).
        confidence_threshold: Minimum confidence score (0.0-1.0) to accept entities.
        normalize_units: Convert units to standard format (e.g., all lengths to mm).
        pattern_file: Optional path to custom regex pattern definitions.
        oem_dictionary_path: Optional path to file containing OEM manufacturer names.
        regex_part_number_confidence: Confidence score for regex-extracted part numbers.
        regex_dimension_confidence: Confidence score for regex-extracted dimensions.
        regex_weight_confidence: Confidence score for regex-extracted weights.
        regex_thread_confidence: Confidence score for regex-extracted thread specs.
        regex_material_confidence: Confidence score for regex-extracted materials.
        regex_finish_confidence: Confidence score for regex-extracted surface finishes.
        regex_tolerance_confidence: Confidence score for regex-extracted tolerances.
        spacy_confidence: Confidence score for spaCy-extracted entities.
        enable_entity_types: List of entity types to extract (empty = all types).
        max_llm_retries: Maximum retry attempts for LLM API failures.
        llm_timeout: Timeout in seconds for LLM API calls.
        spacy_model: spaCy model name to use (default: blank English).
        fuzzy_match_threshold: Similarity threshold (0.0-1.0) for text block matching.
        spatial_dedup_iou_threshold: IoU threshold for spatial deduplication.
    """

    use_regex: bool = True
    use_spacy: bool = True
    use_llm: bool = False
    confidence_threshold: float = 0.80
    normalize_units: bool = True
    pattern_file: Optional[str] = None
    oem_dictionary_path: Optional[str] = None

    # Configurable confidence scores
    regex_part_number_confidence: float = 0.90
    regex_dimension_confidence: float = 0.85
    regex_weight_confidence: float = 0.85
    regex_thread_confidence: float = 0.90
    regex_material_confidence: float = 0.75
    regex_finish_confidence: float = 0.80
    regex_tolerance_confidence: float = 0.85
    spacy_confidence: float = 0.85

    # Advanced options
    enable_entity_types: List[EntityType] = field(default_factory=list)
    max_llm_retries: int = 3
    llm_timeout: int = 30
    spacy_model: str = "blank_en"
    fuzzy_match_threshold: float = 0.85
    spatial_dedup_iou_threshold: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        if not 0.0 <= self.fuzzy_match_threshold <= 1.0:
            raise ValueError(
                f"fuzzy_match_threshold must be between 0.0 and 1.0, "
                f"got {self.fuzzy_match_threshold}"
            )

        if not 0.0 <= self.spatial_dedup_iou_threshold <= 1.0:
            raise ValueError(
                f"spatial_dedup_iou_threshold must be between 0.0 and 1.0, "
                f"got {self.spatial_dedup_iou_threshold}"
            )

        if self.max_llm_retries < 0:
            raise ValueError(
                f"max_llm_retries must be non-negative, got {self.max_llm_retries}"
            )

        if self.llm_timeout <= 0:
            raise ValueError(f"llm_timeout must be positive, got {self.llm_timeout}")

        # Warn if OEM dictionary path is provided but doesn't exist
        if self.oem_dictionary_path:
            path = Path(self.oem_dictionary_path)
            if not path.exists():
                logger.warning(
                    f"OEM dictionary path provided but file does not exist: "
                    f"{self.oem_dictionary_path}"
                )


class EntityExtractor:
    """
    Extracts structured entities from OCR text using a cascading approach.

    The extractor uses three layers in sequence:
    1. Regex patterns: Fast, deterministic extraction for well-structured data
    2. spaCy NER: Context-aware extraction for ambiguous cases
    3. LLM enhancement: Optional fallback for missing critical entities

    This class implements the context manager protocol for proper resource cleanup.

    Attributes:
        config: Entity extraction configuration settings.
        llm_gateway: Optional LLM gateway for enhanced extraction.
        patterns: Dictionary of regex patterns for entity types.
        oem_names: Set of known OEM manufacturer names.

    Example:
        >>> config = EntityConfig(use_llm=False)
        >>> with EntityExtractor(config) as extractor:
        ...     result = extractor.extract_entities(ocr_result)
        ...     print(f"Found {len(result.entities)} entities")
    """

    def __init__(
        self,
        config: EntityConfig,
        llm_gateway: Optional[LLMGatewayProtocol] = None,
    ) -> None:
        """
        Initialize entity extractor.

        Args:
            config: Entity extraction configuration.
            llm_gateway: Optional LLM gateway for enhancement.
        """
        self.config = config
        self.llm_gateway = llm_gateway

        # Load patterns
        self.patterns: Dict[str, str] = self._load_patterns()

        # Load OEM dictionary
        self.oem_names: Set[str] = self._load_oem_dictionary()

        # Lazy load spaCy
        self._nlp = None

        # Normalization cache
        self._normalization_cache: Dict[Tuple[EntityType, str], Dict[str, Any]] = {}

        logger.info("EntityExtractor initialized")

    def __enter__(self) -> "EntityExtractor":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Release resources (spaCy model, caches)."""
        if self._nlp is not None:
            # spaCy models don't have explicit cleanup, but we can remove reference
            self._nlp = None
            logger.debug("Released spaCy model reference")

        self._normalization_cache.clear()
        logger.debug("Cleared normalization cache")

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """
        Check if optional dependencies are installed.

        Returns:
            Dictionary mapping dependency names to availability status.
        """
        dependencies = {}

        try:
            import spacy

            dependencies["spacy"] = True
        except ImportError:
            dependencies["spacy"] = False

        return dependencies

    def extract_entities(self, ocr_result: OCRResult) -> EntityExtractionResult:
        """
        Extracts all entities from OCR text using multi-layer approach.

        This method orchestrates the three-layer extraction process:
        1. Applies regex patterns to all text blocks
        2. Uses spaCy NER on full text (if enabled)
        3. Calls LLM for missing critical entities (if enabled and needed)

        Args:
            ocr_result: OCR extraction result containing text blocks and layout
                regions.

        Returns:
            EntityExtractionResult containing:
                - List of all extracted entities
                - Extracted title block data (if found)
                - Extraction statistics (counts, confidence scores)
                - Placeholder for validation report (added by validator)

        Example:
            >>> result = extractor.extract_entities(ocr_result)
            >>> part_numbers = [e for e in result.entities
            ...                 if e.entity_type == EntityType.PART_NUMBER]
        """
        all_entities: List[Entity] = []
        extraction_methods: Dict[str, int] = {"regex": 0, "spacy": 0, "llm": 0}

        # Layer 1: Regex extraction
        if self.config.use_regex:
            regex_entities = self._extract_with_regex(ocr_result)
            all_entities.extend(regex_entities)
            extraction_methods["regex"] = len(regex_entities)
            logger.debug(f"Regex extraction: {len(regex_entities)} entities")

        # Layer 2: spaCy extraction
        if self.config.use_spacy:
            spacy_entities = self._extract_with_spacy(ocr_result)
            # Avoid duplicates
            spacy_entities = self._deduplicate_entities(spacy_entities, all_entities)
            all_entities.extend(spacy_entities)
            extraction_methods["spacy"] = len(spacy_entities)
            logger.debug(f"spaCy extraction: {len(spacy_entities)} entities")

        # Extract title block
        title_block = self.extract_title_block(ocr_result)

        # Check for missing critical entities
        missing_critical = self._identify_missing_critical(all_entities)

        # Layer 3: LLM enhancement (optional)
        if self.config.use_llm and self.llm_gateway and missing_critical:
            logger.debug(f"Using LLM for missing entities: {missing_critical}")
            llm_entities = self._extract_with_llm(
                ocr_result.full_text, missing_critical
            )
            all_entities.extend(llm_entities)
            extraction_methods["llm"] = len(llm_entities)

        # Calculate statistics
        stats = self._calculate_statistics(all_entities, extraction_methods)

        result = EntityExtractionResult(
            entities=all_entities,
            title_block=title_block,
            extraction_statistics=stats,
            validation_report=None,  # Will be added by validator
        )

        logger.info(
            f"Entity extraction complete: {len(all_entities)} entities, "
            f"avg confidence: {stats.average_confidence:.2f}"
        )

        return result

    def _extract_with_regex(self, ocr_result: OCRResult) -> List[Entity]:
        """
        Performs fast regex-based entity extraction (Layer 1).

        Applies predefined regex patterns to extract:
        - Part numbers (various formats: PN-123, 12345-AB, etc.)
        - Dimensions (with units: mm, cm, m, in, inch)
        - Weights (kg, g, lbs, oz)
        - Thread specifications (M10x1.5, #8-32 UNC, etc.)
        - Materials (Steel AISI 304, Aluminum 6061, etc.)
        - Surface finishes (Ra 3.2, N7, etc.)
        - Tolerances (±0.1 mm, ±0.005 in, etc.)

        Args:
            ocr_result: OCR result containing text blocks with content and
                bounding boxes.

        Returns:
            List of Entity objects with configurable confidence scores.

        Note:
            Regex extraction is deterministic but may produce false positives.
            Confidence scores are user-configurable via EntityConfig.
        """
        entities: List[Entity] = []

        for text_block in ocr_result.text_blocks:
            text = text_block.content

            # Part numbers
            if self._is_entity_type_enabled(EntityType.PART_NUMBER):
                part_numbers = extract_part_number_candidates(text)
                for pn in part_numbers:
                    entities.append(
                        self._create_entity(
                            EntityType.PART_NUMBER,
                            pn,
                            pn,
                            self.config.regex_part_number_confidence,
                            "regex",
                            text_block,
                        )
                    )

            # Dimensions (using proper Unicode for diameter symbols)
            if self._is_entity_type_enabled(EntityType.DIMENSION):
                dim_pattern = (
                    r"\b(?:Ø|ø|⌀)?\s*\d+\.?\d*\s*(?:[±]?\s*\d*\.?\d*)?\s*"
                    r'(?:mm|cm|m|in|inch|")\b'
                )
                dimensions = re.findall(dim_pattern, text)
                for dim in dimensions:
                    dim_value = extract_dimension_value(dim)
                    if dim_value:
                        entities.append(
                            self._create_entity(
                                EntityType.DIMENSION,
                                dim,
                                dim,
                                self.config.regex_dimension_confidence,
                                "regex",
                                text_block,
                                normalized=dim_value,
                            )
                        )

            # Weight
            if self._is_entity_type_enabled(EntityType.WEIGHT):
                weight_pattern = r"\b\d+\.?\d*\s*(?:kg|g|lbs|oz)\b"
                weights = re.findall(weight_pattern, text, re.IGNORECASE)
                for weight in weights:
                    entities.append(
                        self._create_entity(
                            EntityType.WEIGHT,
                            weight,
                            weight,
                            self.config.regex_weight_confidence,
                            "regex",
                            text_block,
                        )
                    )

            # Thread specifications
            if self._is_entity_type_enabled(EntityType.THREAD_SPEC):
                thread_pattern = (
                    r"\b(?:M\d+[xX]?\d*\.?\d*|#?\d+[-/]\d+\s*(?:UNC|UNF))\b"
                )
                threads = re.findall(thread_pattern, text)
                for thread in threads:
                    entities.append(
                        self._create_entity(
                            EntityType.THREAD_SPEC,
                            thread,
                            thread,
                            self.config.regex_thread_confidence,
                            "regex",
                            text_block,
                        )
                    )

            # Material (case-insensitive matching with word boundaries)
            if self._is_entity_type_enabled(EntityType.MATERIAL):
                material_pattern = (
                    r"\b(?:steel|aluminum|brass|stainless|titanium|plastic)\s+"
                    r"[A-Z0-9\-]+\b"
                )
                materials = re.findall(material_pattern, text, re.IGNORECASE)
                for material in materials:
                    entities.append(
                        self._create_entity(
                            EntityType.MATERIAL,
                            material,
                            material,
                            self.config.regex_material_confidence,
                            "regex",
                            text_block,
                        )
                    )

            # Surface finish
            if self._is_entity_type_enabled(EntityType.SURFACE_FINISH):
                finish_pattern = r"\b(?:Ra\s*\d+\.?\d*|N\d+)\b"
                finishes = re.findall(finish_pattern, text)
                for finish in finishes:
                    entities.append(
                        self._create_entity(
                            EntityType.SURFACE_FINISH,
                            finish,
                            finish,
                            self.config.regex_finish_confidence,
                            "regex",
                            text_block,
                        )
                    )

            # Tolerance
            if self._is_entity_type_enabled(EntityType.TOLERANCE):
                tolerance_pattern = r"[±]\s*\d+\.?\d*\s*(?:mm|cm|m|in)?\b"
                tolerances = re.findall(tolerance_pattern, text)
                for tol in tolerances:
                    entities.append(
                        self._create_entity(
                            EntityType.TOLERANCE,
                            tol,
                            tol,
                            self.config.regex_tolerance_confidence,
                            "regex",
                            text_block,
                        )
                    )

        return entities

    def _extract_with_spacy(self, ocr_result: OCRResult) -> List[Entity]:
        """
        Performs spaCy-based entity extraction with EntityRuler (Layer 2).

        Uses spaCy's EntityRuler to extract:
        - OEM manufacturer names from predefined dictionary
        - Other context-aware entities (currently limited)

        The spaCy model is lazy-loaded on first use and cached for subsequent
        calls.

        Args:
            ocr_result: OCR result containing full text and individual text
                blocks.

        Returns:
            List of Entity objects extracted by spaCy. Returns empty list if:
            - spaCy is not installed
            - spaCy initialization fails
            - No entities are found

        Note:
            This method maps extracted entities back to their source text blocks
            using enhanced matching (exact, then fuzzy).
        """
        entities: List[Entity] = []

        # Initialize spaCy if not loaded
        if self._nlp is None:
            try:
                import spacy

                if self.config.spacy_model == "blank_en":
                    self._nlp = spacy.blank("en")
                else:
                    self._nlp = spacy.load(self.config.spacy_model)

                # Add EntityRuler with patterns
                ruler = self._nlp.add_pipe("entity_ruler")
                patterns = [
                    {
                        "label": "OEM",
                        "pattern": [{"LOWER": {"IN": list(self.oem_names)}}],
                    },
                ]
                ruler.add_patterns(patterns)
                logger.info("spaCy EntityRuler initialized successfully")
            except ImportError:
                logger.warning("spaCy not installed, skipping spaCy extraction")
                return entities
            except Exception as e:
                logger.error(f"Failed to initialize spaCy: {e}", exc_info=True)
                return entities

        # Process full text
        doc = self._nlp(ocr_result.full_text)

        # Extract OEM names
        if self._is_entity_type_enabled(EntityType.OEM):
            for ent in doc.ents:
                if ent.label_ == "OEM":
                    # Find which text block this came from (enhanced matching)
                    text_block = self._find_text_block_for_text(
                        ent.text, ocr_result.text_blocks
                    )
                    if text_block:
                        entities.append(
                            self._create_entity(
                                EntityType.OEM,
                                ent.text,
                                ent.text,
                                self.config.spacy_confidence,
                                "spacy",
                                text_block,
                            )
                        )

        return entities

    def _extract_with_llm(
        self, text: str, failed_types: List[EntityType]
    ) -> List[Entity]:
        """
        Performs LLM-based entity extraction for missing critical entities
        (Layer 3).

        This method is called only when:
        1. LLM enhancement is enabled (config.use_llm=True)
        2. An LLM gateway is configured
        3. Critical entities (PART_NUMBER, OEM) are missing after regex/spaCy
           extraction

        Implements retry logic for transient API failures.

        Args:
            text: Full concatenated text from OCR result.
            failed_types: List of entity types that weren't found in previous
                layers.

        Returns:
            List of Entity objects extracted by LLM. Returns empty list on
            failure.

        Warning:
            This method incurs LLM API costs. Use sparingly (5-10% of drawings).
        """
        if not self.llm_gateway:
            return []

        entities: List[Entity] = []

        for attempt in range(self.config.max_llm_retries):
            try:
                # Call LLM for entity extraction
                from ..llm.llm_gateway import UseCaseType

                llm_entities = self.llm_gateway.extract_entities_llm(
                    text=text,
                    context="Technical engineering drawing",
                    entity_types=[et.value for et in failed_types],
                )

                entities.extend(llm_entities)
                break  # Success, exit retry loop

            except Exception as e:
                if attempt < self.config.max_llm_retries - 1:
                    logger.warning(
                        f"LLM entity extraction failed (attempt {attempt + 1}/"
                        f"{self.config.max_llm_retries}): {e}"
                    )
                else:
                    logger.error(
                        f"LLM entity extraction failed after "
                        f"{self.config.max_llm_retries} attempts: {e}",
                        exc_info=True,
                    )

        return entities

    def extract_title_block(self, ocr_result: OCRResult) -> Optional[TitleBlock]:
        """
        Extracts structured title block data from designated layout region.

        Searches for a layout region marked as 'title_block' and extracts:
        - Part number
        - OEM manufacturer name
        - Scale (e.g., 1:1, 2:1)
        - Date (various formats: MM/DD/YYYY, DD-MM-YYYY)
        - Revision (e.g., A, Rev B, R01)

        Args:
            ocr_result: OCR result with layout regions and text blocks.

        Returns:
            TitleBlock object with extracted fields (may have None values), or
            None if no title block region is found or it contains no text.

        Note:
            Title block extraction assumes bottom-right positioning and uses
            simple regex patterns. Complex layouts may require LLM enhancement.
        """
        # Find title block region
        title_block_region = None
        for region in ocr_result.layout_regions:
            if region.region_type == "title_block":
                title_block_region = region
                break

        if not title_block_region:
            return None

        # Get text blocks in title block region
        title_text_blocks = [
            tb for tb in ocr_result.text_blocks if tb.region_type == "title_block"
        ]

        if not title_text_blocks:
            return None

        # Extract title block fields
        title_block = TitleBlock()

        combined_text = " ".join(tb.content for tb in title_text_blocks)

        # Extract part number
        part_numbers = extract_part_number_candidates(combined_text)
        if part_numbers:
            title_block.part_number = part_numbers[0]

        # Extract OEM
        for oem_name in self.oem_names:
            if oem_name.lower() in combined_text.lower():
                title_block.oem = oem_name
                break

        # Extract scale
        scale_pattern = r"(?:SCALE|Scale)[:\s]*(\d+:\d+|1:\d+)"
        scale_match = re.search(scale_pattern, combined_text)
        if scale_match:
            title_block.scale = scale_match.group(1)

        # Extract date (simplified)
        date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        date_match = re.search(date_pattern, combined_text)
        if date_match:
            title_block.date = date_match.group(0)

        # Extract revision
        rev_pattern = r"(?:REV|Rev|Revision)[:\s]*([A-Z0-9]+)"
        rev_match = re.search(rev_pattern, combined_text)
        if rev_match:
            title_block.revision = rev_match.group(1)

        return title_block

    def _create_entity(
        self,
        entity_type: EntityType,
        value: str,
        original_text: str,
        confidence: float,
        method: str,
        text_block: TextBlock,
        normalized: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """
        Creates an Entity object with proper normalization and metadata.

        Args:
            entity_type: Type of entity being created (from EntityType enum).
            value: Extracted entity value (may be cleaned/processed).
            original_text: Raw text exactly as it appeared in the source.
            confidence: Confidence score (0.0-1.0) for this extraction.
            method: Extraction method used ('regex', 'spacy', or 'llm').
            text_block: Source text block containing this entity.
            normalized: Optional pre-computed normalized value dictionary.

        Returns:
            Entity object with unique ID, bounding box, and normalized values.

        Note:
            If normalized is None, calls _normalize_entity() to compute it.
        """
        # Normalize value if needed (with caching)
        if normalized is None:
            cache_key = (entity_type, value)
            if cache_key in self._normalization_cache:
                normalized = self._normalization_cache[cache_key]
            else:
                normalized = self._normalize_entity(entity_type, value)
                self._normalization_cache[cache_key] = normalized

        return Entity(
            entity_id=generate_unique_id("ENT"),
            entity_type=entity_type,
            value=value,
            original_text=original_text,
            normalized_value=normalized,
            confidence=confidence,
            extraction_method=method,
            source_text_id=text_block.text_id,
            bbox=text_block.bbox,
        )

    def _normalize_entity(self, entity_type: EntityType, value: str) -> Dict[str, Any]:
        """
        Normalizes entity values to standard formats for consistent storage.

        Normalization rules by entity type:
        - DIMENSION: Extracts numeric value, unit, tolerance; converts to mm
          if configured
        - WEIGHT: Extracts numeric value and unit (lowercase)
        - Other types: Stores raw value only

        Args:
            entity_type: Type of entity to normalize.
            value: Raw extracted value string.

        Returns:
            Dictionary containing:
            - 'raw': Original value (always present)
            - Additional keys depending on entity_type (value, unit,
              tolerance, etc.)

        Example:
            >>> result = extractor._normalize_entity(
            ...     EntityType.DIMENSION, "25.4mm ±0.1"
            ... )
            >>> print(result)
            {'raw': '25.4mm ±0.1', 'value': 25.4, 'unit': 'mm',
             'tolerance': 0.1}
        """
        normalized: Dict[str, Any] = {"raw": value}

        if entity_type == EntityType.DIMENSION:
            dim = extract_dimension_value(value)
            if dim:
                normalized.update(dim)
                # Convert to standard unit (mm) if configured
                if self.config.normalize_units and dim.get("unit"):
                    try:
                        if dim["unit"] != "mm":
                            normalized["value_mm"] = convert_unit(
                                dim["value"], dim["unit"], "mm"
                            )
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to convert dimension '{value}' to mm: {e}"
                        )

        elif entity_type == EntityType.WEIGHT:
            # Extract numeric value and unit
            match = re.match(r"(\d+\.?\d*)\s*([a-zA-Z]+)", value)
            if match:
                try:
                    num_val, unit = match.groups()
                    normalized["value"] = float(num_val)
                    normalized["unit"] = unit.lower()
                except ValueError as e:
                    logger.warning(f"Failed to normalize weight '{value}': {e}")

        return normalized

    def _load_patterns(self) -> Dict[str, str]:
        """
        Loads regex patterns from file or returns defaults.

        Returns:
            Dictionary mapping pattern names to regex strings. Currently
            returns empty dict as patterns are embedded in extraction methods.

        Todo:
            Implement pattern file loading when config.pattern_file is
            provided to allow external pattern customization.
        """
        # Pattern file loading not yet implemented
        # Patterns are currently hardcoded in _extract_with_regex for simplicity
        return {}

    def _load_oem_dictionary(self) -> Set[str]:
        """
        Loads OEM manufacturer names from dictionary file or uses defaults.

        Returns:
            Set of OEM manufacturer names (case-preserved). Includes 11
            built-in names plus any loaded from config.oem_dictionary_path.

        Note:
            Logs warning if file path is provided but loading fails.
        """
        oem_names = {
            "SKF",
            "Timken",
            "NSK",
            "NTN",
            "FAG",
            "INA",
            "Schaeffler",
            "KOYO",
            "NACHI",
            "THK",
            "IKO",
        }

        if (
            self.config.oem_dictionary_path
            and Path(self.config.oem_dictionary_path).exists()
        ):
            try:
                with open(self.config.oem_dictionary_path, "r") as f:
                    for line in f:
                        name = line.strip()
                        if name:
                            oem_names.add(name)
                logger.info(
                    f"Loaded {len(oem_names) - 11} additional OEM names from "
                    f"dictionary"
                )
            except (OSError, IOError) as e:
                logger.warning(f"Failed to load OEM dictionary: {e}", exc_info=True)

        return oem_names

    def _deduplicate_entities(
        self, new_entities: List[Entity], existing_entities: List[Entity]
    ) -> List[Entity]:
        """
        Removes duplicate entities using type, value, and spatial matching.

        This method checks for duplicates by:
        1. Type and value matching (case-insensitive)
        2. Spatial proximity (using IoU if bounding boxes overlap significantly)

        Args:
            new_entities: List of newly extracted entities to check.
            existing_entities: List of already-accepted entities.

        Returns:
            Filtered list containing only entities that don't match any
            existing entity.

        Note:
            Spatial deduplication prevents discarding valid but spatially
            distinct entities with the same value (e.g., repeated dimensions).
        """
        unique: List[Entity] = []

        for new_ent in new_entities:
            is_duplicate = False
            for exist_ent in existing_entities:
                # Check type and value match
                if (
                    new_ent.entity_type == exist_ent.entity_type
                    and new_ent.value.lower() == exist_ent.value.lower()
                ):
                    # If both have bounding boxes, check spatial overlap
                    if new_ent.bbox and exist_ent.bbox:
                        iou = self._calculate_iou(new_ent.bbox, exist_ent.bbox)
                        if iou >= self.config.spatial_dedup_iou_threshold:
                            is_duplicate = True
                            break
                    else:
                        # No bbox info, assume duplicate
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique.append(new_ent)

        return unique

    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box.
            bbox2: Second bounding box.

        Returns:
            IoU score between 0.0 (no overlap) and 1.0 (complete overlap).
        """
        # Calculate intersection
        x1_inter = max(bbox1.x, bbox2.x)
        y1_inter = max(bbox1.y, bbox2.y)
        x2_inter = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2_inter = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        bbox1_area = bbox1.width * bbox1.height
        bbox2_area = bbox2.width * bbox2.height
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _identify_missing_critical(self, entities: List[Entity]) -> List[EntityType]:
        """
        Identifies critical entity types that are missing from extraction
        results.

        Critical entities are defined as:
        - PART_NUMBER: Required for database indexing
        - OEM: Required for manufacturer attribution

        Args:
            entities: List of extracted entities to check.

        Returns:
            List of EntityType values that are critical but not present in
            entities.

        Note:
            Used to determine whether LLM enhancement should be triggered.
        """
        found_types = {e.entity_type for e in entities}
        critical_types = {EntityType.PART_NUMBER, EntityType.OEM}

        missing = [et for et in critical_types if et not in found_types]
        return missing

    def _find_text_block_for_text(
        self, text: str, text_blocks: List[TextBlock]
    ) -> Optional[TextBlock]:
        """
        Finds the text block containing a given text string using enhanced
        matching.

        Uses a three-step approach:
        1. Exact substring match
        2. Fuzzy matching (for OCR errors)
        3. Returns None if no match found

        Args:
            text: String to search for within text blocks.
            text_blocks: List of text blocks to search.

        Returns:
            First TextBlock whose content contains or closely matches the
            search text, or None if not found.

        Note:
            Fuzzy matching threshold is configurable via
            config.fuzzy_match_threshold.
        """
        # Step 1: Try exact match first
        for tb in text_blocks:
            if text in tb.content:
                return tb

        # Step 2: Try fuzzy matching
        best_match = None
        best_ratio = 0.0

        for tb in text_blocks:
            ratio = SequenceMatcher(None, text, tb.content).ratio()
            if ratio > best_ratio and ratio >= self.config.fuzzy_match_threshold:
                best_ratio = ratio
                best_match = tb

        if best_match:
            logger.debug(
                f"Fuzzy matched text '{text[:30]}...' with ratio {best_ratio:.2f}"
            )
            return best_match

        # Step 3: No match found
        logger.debug(f"No text block found for text: '{text[:30]}...'")
        return None

    def _is_entity_type_enabled(self, entity_type: EntityType) -> bool:
        """
        Check if a specific entity type is enabled for extraction.

        Args:
            entity_type: Entity type to check.

        Returns:
            True if entity type should be extracted, False otherwise.

        Note:
            If config.enable_entity_types is empty, all types are enabled.
        """
        if not self.config.enable_entity_types:
            return True
        return entity_type in self.config.enable_entity_types

    def _calculate_statistics(
        self, entities: List[Entity], extraction_methods: Dict[str, int]
    ) -> ExtractionStats:
        """
        Calculates extraction statistics for reporting and quality assessment.

        Args:
            entities: List of all extracted entities.
            extraction_methods: Dictionary mapping method names ('regex',
                'spacy', 'llm') to count of entities extracted by each method.

        Returns:
            ExtractionStats containing:
            - total_entities: Total count
            - entities_by_type: Dict mapping entity type names to counts
            - entities_by_method: Copy of input extraction_methods dict
            - average_confidence: Mean confidence across all entities

        Note:
            Returns 0.0 average confidence if entities list is empty.
        """
        entities_by_type: Dict[str, int] = {}
        for entity in entities:
            type_name = entity.entity_type.value
            entities_by_type[type_name] = entities_by_type.get(type_name, 0) + 1

        avg_confidence = (
            sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        )

        return ExtractionStats(
            total_entities=len(entities),
            entities_by_type=entities_by_type,
            entities_by_method=extraction_methods,
            average_confidence=avg_confidence,
        )
