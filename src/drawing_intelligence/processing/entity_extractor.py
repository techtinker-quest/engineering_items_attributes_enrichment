"""
Entity extraction module for the Drawing Intelligence System.

Multi-layer extraction: Regex → spaCy → LLM (optional).
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path

from ..models.data_structures import (
    OCRResult,
    Entity,
    EntityType,
    TitleBlock,
    EntityExtractionResult,
    ExtractionStats,
    TextBlock,
)
from ..utils.file_utils import generate_unique_id
from ..utils.text_utils import (
    extract_dimension_value,
    extract_part_number_candidates,
    normalize_technical_symbols,
)
from ..utils.geometry_utils import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class EntityConfig:
    """
    Configuration for entity extraction.

    Attributes:
        use_regex: Enable regex-based extraction (default: True)
        use_spacy: Enable spaCy-based extraction (default: True)
        use_llm: Enable LLM-based extraction (default: False)
        confidence_threshold: Minimum confidence for entities (default: 0.80)
        normalize_units: Convert units to standard (default: True)
        pattern_file: Path to custom regex patterns (optional)
        oem_dictionary_path: Path to OEM name dictionary (optional)
    """

    use_regex: bool = True
    use_spacy: bool = True
    use_llm: bool = False
    confidence_threshold: float = 0.80
    normalize_units: bool = True
    pattern_file: Optional[str] = None
    oem_dictionary_path: Optional[str] = None


class EntityExtractor:
    """
    Extract structured entities from OCR text.

    Three-layer approach:
    1. Regex patterns (fast, precise)
    2. spaCy EntityRuler (context-aware)
    3. LLM enhancement (optional, for difficult cases)
    """

    def __init__(self, config: EntityConfig, llm_gateway: Optional[Any] = None):
        """
        Initialize entity extractor.

        Args:
            config: Entity extraction configuration
            llm_gateway: Optional LLM gateway for enhancement
        """
        self.config = config
        self.llm_gateway = llm_gateway

        # Load patterns
        self.patterns = self._load_patterns()

        # Load OEM dictionary
        self.oem_names = self._load_oem_dictionary()

        # Lazy load spaCy
        self._nlp = None

        logger.info("EntityExtractor initialized")

    def extract_entities(self, ocr_result: OCRResult) -> EntityExtractionResult:
        """
        Extract all entities from OCR text using multi-layer approach.

        Args:
            ocr_result: OCR extraction result

        Returns:
            EntityExtractionResult with entities and statistics
        """
        all_entities = []
        extraction_methods = {"regex": 0, "spacy": 0, "llm": 0}

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
        Layer 1: Fast regex-based extraction.

        Args:
            ocr_result: OCR result

        Returns:
            List of extracted entities
        """
        entities = []

        for text_block in ocr_result.text_blocks:
            text = text_block.content

            # Part numbers
            part_numbers = extract_part_number_candidates(text)
            for pn in part_numbers:
                entities.append(
                    self._create_entity(
                        EntityType.PART_NUMBER, pn, pn, 0.9, "regex", text_block
                    )
                )

            # Dimensions
            dim_pattern = r'(?:Ø|ø|⌀)?\s*\d+\.?\d*\s*(?:[±]?\s*\d*\.?\d*)?\s*(?:mm|cm|m|in|inch|")'
            dimensions = re.findall(dim_pattern, text)
            for dim in dimensions:
                dim_value = extract_dimension_value(dim)
                if dim_value:
                    entities.append(
                        self._create_entity(
                            EntityType.DIMENSION,
                            dim,
                            dim,
                            0.85,
                            "regex",
                            text_block,
                            normalized=dim_value,
                        )
                    )

            # Weight
            weight_pattern = r"\d+\.?\d*\s*(?:kg|g|lbs|oz)"
            weights = re.findall(weight_pattern, text, re.IGNORECASE)
            for weight in weights:
                entities.append(
                    self._create_entity(
                        EntityType.WEIGHT, weight, weight, 0.85, "regex", text_block
                    )
                )

            # Thread specifications
            thread_pattern = r"(?:M\d+[xX]?\d*\.?\d*|#?\d+[-/]\d+\s*(?:UNC|UNF))"
            threads = re.findall(thread_pattern, text)
            for thread in threads:
                entities.append(
                    self._create_entity(
                        EntityType.THREAD_SPEC, thread, thread, 0.9, "regex", text_block
                    )
                )

            # Material (simplified patterns)
            material_pattern = (
                r"(?:Steel|Aluminum|Brass|Stainless|Titanium|Plastic)\s+[A-Z0-9\-]+"
            )
            materials = re.findall(material_pattern, text, re.IGNORECASE)
            for material in materials:
                entities.append(
                    self._create_entity(
                        EntityType.MATERIAL,
                        material,
                        material,
                        0.75,
                        "regex",
                        text_block,
                    )
                )

            # Surface finish
            finish_pattern = r"Ra\s*\d+\.?\d*|N\d+"
            finishes = re.findall(finish_pattern, text)
            for finish in finishes:
                entities.append(
                    self._create_entity(
                        EntityType.SURFACE_FINISH,
                        finish,
                        finish,
                        0.8,
                        "regex",
                        text_block,
                    )
                )

            # Tolerance
            tolerance_pattern = r"[±]\s*\d+\.?\d*\s*(?:mm|cm|m|in)?"
            tolerances = re.findall(tolerance_pattern, text)
            for tol in tolerances:
                entities.append(
                    self._create_entity(
                        EntityType.TOLERANCE, tol, tol, 0.85, "regex", text_block
                    )
                )

        return entities

    def _extract_with_spacy(self, ocr_result: OCRResult) -> List[Entity]:
        """
        Layer 2: spaCy-based extraction with EntityRuler.

        Args:
            ocr_result: OCR result

        Returns:
            List of extracted entities
        """
        entities = []

        # Initialize spaCy if not loaded
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.blank("en")
                # Add EntityRuler with patterns
                ruler = self._nlp.add_pipe("entity_ruler")
                # Add patterns (simplified for demonstration)
                patterns = [
                    {
                        "label": "OEM",
                        "pattern": [{"LOWER": {"IN": list(self.oem_names)}}],
                    },
                ]
                ruler.add_patterns(patterns)
            except ImportError:
                logger.warning("spaCy not installed, skipping spaCy extraction")
                return entities
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy: {e}")
                return entities

        # Process full text
        doc = self._nlp(ocr_result.full_text)

        # Extract OEM names
        for ent in doc.ents:
            if ent.label_ == "OEM":
                # Find which text block this came from
                text_block = self._find_text_block_for_text(
                    ent.text, ocr_result.text_blocks
                )
                if text_block:
                    entities.append(
                        self._create_entity(
                            EntityType.OEM,
                            ent.text,
                            ent.text,
                            0.85,
                            "spacy",
                            text_block,
                        )
                    )

        return entities

    def _extract_with_llm(
        self, text: str, failed_types: List[EntityType]
    ) -> List[Entity]:
        """
        Layer 3: LLM-based extraction for failed cases.

        Args:
            text: Full text content
            failed_types: Entity types that weren't found

        Returns:
            List of extracted entities
        """
        if not self.llm_gateway:
            return []

        entities = []

        try:
            # Call LLM for entity extraction
            from ..llm.llm_gateway import UseCaseType

            llm_entities = self.llm_gateway.extract_entities_llm(
                text=text,
                context="Technical engineering drawing",
                entity_types=[et.value for et in failed_types],
            )

            entities.extend(llm_entities)

        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")

        return entities

    def extract_title_block(self, ocr_result: OCRResult) -> Optional[TitleBlock]:
        """
        Extract structured title block data.

        Args:
            ocr_result: OCR result with layout regions

        Returns:
            TitleBlock object or None
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
        normalized: Optional[Dict] = None,
    ) -> Entity:
        """
        Create Entity object with proper normalization.

        Args:
            entity_type: Type of entity
            value: Extracted value
            original_text: Original text
            confidence: Confidence score
            method: Extraction method
            text_block: Source text block
            normalized: Optional normalized value dict

        Returns:
            Entity object
        """
        # Normalize value if needed
        if normalized is None:
            normalized = self._normalize_entity(entity_type, value)

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
        Normalize entity value to standard format.

        Args:
            entity_type: Type of entity
            value: Raw value

        Returns:
            Dictionary with normalized values
        """
        normalized = {"raw": value}

        if entity_type == EntityType.DIMENSION:
            dim = extract_dimension_value(value)
            if dim:
                normalized.update(dim)
                # Convert to standard unit (mm) if configured
                if self.config.normalize_units and dim.get("unit"):
                    from ..utils.text_utils import convert_unit

                    try:
                        if dim["unit"] != "mm":
                            normalized["value_mm"] = convert_unit(
                                dim["value"], dim["unit"], "mm"
                            )
                    except:
                        pass

        elif entity_type == EntityType.WEIGHT:
            # Extract numeric value and unit
            import re

            match = re.match(r"(\d+\.?\d*)\s*([a-zA-Z]+)", value)
            if match:
                num_val, unit = match.groups()
                normalized["value"] = float(num_val)
                normalized["unit"] = unit.lower()

        return normalized

    def _load_patterns(self) -> Dict[str, str]:
        """Load regex patterns from file or use defaults."""
        # Use built-in patterns for now
        return {}

    def _load_oem_dictionary(self) -> set:
        """Load OEM names from dictionary file."""
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
            except Exception as e:
                logger.warning(f"Failed to load OEM dictionary: {e}")

        return oem_names

    def _deduplicate_entities(
        self, new_entities: List[Entity], existing_entities: List[Entity]
    ) -> List[Entity]:
        """Remove entities that are duplicates of existing ones."""
        unique = []

        for new_ent in new_entities:
            is_duplicate = False
            for exist_ent in existing_entities:
                if (
                    new_ent.entity_type == exist_ent.entity_type
                    and new_ent.value.lower() == exist_ent.value.lower()
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(new_ent)

        return unique

    def _identify_missing_critical(self, entities: List[Entity]) -> List[EntityType]:
        """Identify critical entity types that are missing."""
        found_types = {e.entity_type for e in entities}
        critical_types = {EntityType.PART_NUMBER, EntityType.OEM}

        missing = [et for et in critical_types if et not in found_types]
        return missing

    def _find_text_block_for_text(
        self, text: str, text_blocks: List[TextBlock]
    ) -> Optional[TextBlock]:
        """Find which text block contains the given text."""
        for tb in text_blocks:
            if text in tb.content:
                return tb
        return None

    def _calculate_statistics(
        self, entities: List[Entity], extraction_methods: Dict[str, int]
    ) -> ExtractionStats:
        """Calculate extraction statistics."""
        entities_by_type = {}
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
