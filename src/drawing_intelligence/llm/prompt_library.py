"""
Prompt library for LLM use cases.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """
    Prompt template definition.

    Attributes:
        name: Template name
        version: Version string
        template: Prompt template text with {variables}
        variables: List of variable names
        token_budget: Estimated token budget
        recommended_model: Recommended model for this prompt
        metadata: Additional metadata
    """

    name: str
    version: str
    template: str
    variables: List[str]
    token_budget: int
    recommended_model: str
    metadata: Dict[str, Any]


class PromptLibrary:
    """
    Manage versioned prompts for different use cases.

    Provides:
    - Prompt templates with variable substitution
    - Version control
    - Token budget tracking
    - Model recommendations
    """

    def __init__(self, prompts_dir: str = "config/prompts"):
        """
        Initialize prompt library.

        Args:
            prompts_dir: Directory containing prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._prompts: Dict[str, Dict[str, PromptTemplate]] = {}

        # Create directory if it doesn't exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Load built-in prompts
        self._load_builtin_prompts()

        # Load custom prompts from directory
        if self.prompts_dir.exists():
            self._load_prompts_from_dir()

        logger.info(f"PromptLibrary initialized with {len(self._prompts)} use cases")

    def get_prompt(self, use_case: str, version: str = "latest") -> PromptTemplate:
        """
        Get prompt template for use case.

        Args:
            use_case: Use case name (e.g., 'drawing_assessment')
            version: Version string or 'latest'

        Returns:
            PromptTemplate

        Raises:
            KeyError: If use case or version not found
        """
        if use_case not in self._prompts:
            raise KeyError(f"Use case not found: {use_case}")

        versions = self._prompts[use_case]

        if version == "latest":
            # Get highest version
            version = max(versions.keys())

        if version not in versions:
            raise KeyError(f"Version {version} not found for {use_case}")

        return versions[version]

    def render_prompt(self, template: PromptTemplate, context: Dict[str, Any]) -> str:
        """
        Render template with variables.

        Args:
            template: PromptTemplate
            context: Dictionary with variable values

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If required variable missing
        """
        # Check all required variables are present
        missing = [var for var in template.variables if var not in context]
        if missing:
            raise KeyError(f"Missing required variables: {missing}")

        # Render template
        try:
            rendered = template.template.format(**context)
            return rendered
        except KeyError as e:
            raise KeyError(f"Template rendering failed: {e}")

    def list_available_prompts(self) -> Dict[str, List[str]]:
        """
        List all prompts by use case.

        Returns:
            Dictionary mapping use case to list of versions
        """
        return {
            use_case: list(versions.keys())
            for use_case, versions in self._prompts.items()
        }

    def add_prompt(self, prompt: PromptTemplate):
        """
        Add a custom prompt to the library.

        Args:
            prompt: PromptTemplate to add
        """
        if prompt.name not in self._prompts:
            self._prompts[prompt.name] = {}

        self._prompts[prompt.name][prompt.version] = prompt
        logger.info(f"Added prompt: {prompt.name} v{prompt.version}")

    def _load_builtin_prompts(self):
        """Load built-in prompt templates."""

        # Drawing Assessment Prompt
        drawing_assessment = PromptTemplate(
            name="drawing_assessment",
            version="v1.2",
            template="""You are an expert at analyzing technical engineering drawings.

Analyze this drawing and provide a quality assessment.

Evaluate the following:
1. Overall Quality (0-10): How clear and readable is the drawing?
2. Complexity (0-10): How complex is the technical content?
3. Text Clarity (0-10): How legible is the text?
4. Shape Clarity (0-10): How clear are the component shapes?

Drawing file: {file_name}

Provide your assessment in this format:
Overall Quality: [score]/10
Complexity: [score]/10
Text Clarity: [score]/10
Shape Clarity: [score]/10
Recommended Pipeline: [baseline/hybrid/llm_enhanced]
Reasoning: [brief explanation]
""",
            variables=["file_name"],
            token_budget=1000,
            recommended_model="claude-3-haiku-20240307",
            metadata={"use_case_type": "DRAWING_ASSESSMENT"},
        )
        self.add_prompt(drawing_assessment)

        # OCR Verification Prompt
        ocr_verification = PromptTemplate(
            name="ocr_verification",
            version="v1.1",
            template="""You are verifying OCR text extracted from a technical drawing.

The OCR system extracted this text:
"{ocr_text}"

This text is from a {region_type} region.

Please:
1. Identify and correct any OCR errors
2. Preserve technical symbols (Ø, ±, °, etc.)
3. Maintain proper formatting
4. List any corrections made

Provide corrected text and corrections list:
Corrected Text: [corrected version]
Corrections: [list of changes]
""",
            variables=["ocr_text", "region_type"],
            token_budget=2000,
            recommended_model="claude-3-sonnet-20240229",
            metadata={"use_case_type": "OCR_VERIFICATION"},
        )
        self.add_prompt(ocr_verification)

        # Entity Extraction Prompt
        entity_extraction = PromptTemplate(
            name="entity_extraction",
            version="v1.0",
            template="""Extract structured entities from this technical drawing text.

Text:
{text}

Context: {context}

Extract these entity types:
{entity_types}

For each entity found, provide:
- Type
- Value
- Confidence (0.0-1.0)

Respond in JSON format:
{{
  "entities": [
    {{"type": "...", "value": "...", "confidence": ...}},
    ...
  ]
}}
""",
            variables=["text", "context", "entity_types"],
            token_budget=4000,
            recommended_model="gpt-4-turbo-2024-04-09",
            metadata={"use_case_type": "ENTITY_EXTRACTION"},
        )
        self.add_prompt(entity_extraction)

        # Shape Validation Prompt
        shape_validation = PromptTemplate(
            name="shape_validation",
            version="v1.0",
            template="""Validate these shape detections from a technical drawing.

Detected shapes:
{detections_json}

Review the detections and:
1. Identify any likely false positives
2. Suggest missing shapes that should be detected
3. Validate component classifications

Provide validation results:
False Positives: [list]
Missing Shapes: [list]
Classification Issues: [list]
Overall Confidence: [0.0-1.0]
""",
            variables=["detections_json"],
            token_budget=1500,
            recommended_model="gpt-4-vision-preview",
            metadata={"use_case_type": "SHAPE_VALIDATION"},
        )
        self.add_prompt(shape_validation)

    def _load_prompts_from_dir(self):
        """Load custom prompts from directory."""
        try:
            for prompt_file in self.prompts_dir.glob("*.txt"):
                # Parse prompt file
                # Format: name_version.txt
                stem = prompt_file.stem
                parts = stem.rsplit("_", 1)

                if len(parts) == 2:
                    name, version = parts
                else:
                    name = stem
                    version = "v1.0"

                # Read template
                template_text = prompt_file.read_text()

                # Extract variables (simple {var} format)
                import re

                variables = re.findall(r"\{(\w+)\}", template_text)

                prompt = PromptTemplate(
                    name=name,
                    version=version,
                    template=template_text,
                    variables=list(set(variables)),
                    token_budget=2000,  # Default
                    recommended_model="gpt-4-turbo",  # Default
                    metadata={},
                )

                self.add_prompt(prompt)

        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")
