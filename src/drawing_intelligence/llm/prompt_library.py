"""Prompt library for LLM use cases.

This module provides a centralized system for managing versioned prompt templates
used across different LLM providers. It supports variable substitution, token
budget tracking, and model recommendations for cost-optimized LLM operations.

The library loads both built-in prompts (hardcoded for core use cases) and
custom prompts from the config/prompts directory (supports .txt and .yaml formats).

Example:
    >>> library = PromptLibrary()
    >>> template = library.get_prompt("drawing_assessment")
    >>> prompt = library.render_prompt(template, {"file_name": "drawing_001.pdf"})

    # Or render directly:
    >>> prompt = library.render("drawing_assessment", {"file_name": "drawing_001.pdf"})
"""

import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from packaging import version

    VERSION_AVAILABLE = True
except ImportError:
    VERSION_AVAILABLE = False

logger = logging.getLogger(__name__)


# Custom Exceptions
class PromptLibraryError(Exception):
    """Base exception for prompt library errors."""

    pass


class PromptNotFoundError(PromptLibraryError):
    """Raised when a requested prompt is not found."""

    def __init__(self, use_case: str, version: Optional[str] = None):
        self.use_case = use_case
        self.version = version
        if version:
            msg = f"Prompt not found: {use_case} (version: {version})"
        else:
            msg = f"Use case not found: {use_case}"
        super().__init__(msg)


class PromptRenderError(PromptLibraryError):
    """Raised when prompt rendering fails."""

    def __init__(
        self, template_name: str, reason: str, missing_vars: Optional[List[str]] = None
    ):
        self.template_name = template_name
        self.reason = reason
        self.missing_vars = missing_vars or []
        msg = f"Failed to render prompt '{template_name}': {reason}"
        if missing_vars:
            msg += f" (missing variables: {missing_vars})"
        super().__init__(msg)


class PromptValidationError(PromptLibraryError):
    """Raised when prompt validation fails."""

    pass


@dataclass(frozen=True)
class PromptTemplate:
    """Prompt template definition with versioning and metadata.

    A structured container for LLM prompt templates that supports variable
    substitution, token budget estimation, and model recommendations.

    Attributes:
        name: Unique identifier for the prompt use case (e.g., 'drawing_assessment').
        version: Semantic version string (e.g., 'v1.2' or '1.2.0').
        template: Prompt text with {variable} placeholders for string formatting.
        variables: List of variable names required for template rendering.
        token_budget: Estimated token count for budget tracking and model selection.
        recommended_model: Model identifier from ModelRegistry for optimal performance.
        metadata: Additional context (e.g., use_case_type, testing_notes).
        deprecated: Whether this prompt version is deprecated.
    """

    name: str
    version: str
    template: str
    variables: List[str]
    token_budget: int
    recommended_model: str
    metadata: Dict[str, Any]
    deprecated: bool = False

    def __post_init__(self):
        """Validate template after initialization."""
        if not self.name or not self.name.strip():
            raise PromptValidationError("Prompt name cannot be empty")
        if not self.version or not self.version.strip():
            raise PromptValidationError("Prompt version cannot be empty")
        if not self.template or not self.template.strip():
            raise PromptValidationError("Prompt template cannot be empty")
        if self.token_budget < 0:
            raise PromptValidationError("Token budget must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the template.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with template data.

        Returns:
            PromptTemplate instance.
        """
        return cls(**data)


class PromptLibrary:
    """Manage versioned prompts for different LLM use cases.

    This class provides a centralized repository for prompt templates with
    version control, variable substitution, token budget tracking, and model
    recommendations. It loads both built-in prompts (for core use cases) and
    custom prompts from the filesystem.

    Supports lazy loading and caching for optimal performance.

    Attributes:
        prompts_dir: Path to directory containing custom prompt files.
        default_token_budget: Default token budget for custom prompts without explicit budget.
        default_model: Default model for custom prompts without explicit model.

    Example:
        >>> library = PromptLibrary("config/prompts")
        >>> template = library.get_prompt("drawing_assessment", "v1.2")
        >>> context = {"file_name": "DWG-001.pdf"}
        >>> rendered = library.render_prompt(template, context)

        # Or use the convenience method:
        >>> rendered = library.render("drawing_assessment", context)
    """

    def __init__(
        self,
        prompts_dir: str = "config/prompts",
        default_token_budget: int = 2000,
        default_model: str = "gpt-4-turbo",
    ) -> None:
        """Initialize prompt library and load all available prompts.

        Creates the prompts directory if it doesn't exist, loads built-in
        prompts, and prepares to lazy-load custom prompts from the directory.

        Args:
            prompts_dir: Relative or absolute path to prompts directory.
            default_token_budget: Default token budget for custom prompts.
            default_model: Default model for custom prompts.

        Raises:
            PromptValidationError: If prompts_dir contains invalid characters.
        """
        # Validate and sanitize prompts directory
        self.prompts_dir = self._validate_path(prompts_dir)
        self.default_token_budget = default_token_budget
        self.default_model = default_model

        self._prompts: Dict[str, Dict[str, PromptTemplate]] = {}
        self._latest_cache: Dict[str, str] = {}  # Cache for latest versions
        self._loaded_files: Set[str] = set()  # Track loaded custom files

        # Create directory if it doesn't exist
        try:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create prompts directory: {e}")

        # Load built-in prompts
        self._load_builtin_prompts()

        logger.info(
            f"PromptLibrary initialized with {len(self._prompts)} built-in use cases. "
            f"Custom prompts dir: {self.prompts_dir}"
        )

    def get_prompt(self, use_case: str, version: str = "latest") -> PromptTemplate:
        """Retrieve prompt template for specified use case and version.

        Lazy-loads custom prompts on first access to improve startup performance.

        Args:
            use_case: Use case identifier (e.g., 'drawing_assessment').
            version: Semantic version string (e.g., 'v1.2', '1.2.0') or 'latest'.

        Returns:
            PromptTemplate matching the specified use case and version.

        Raises:
            PromptNotFoundError: If use_case or version doesn't exist.

        Example:
            >>> template = library.get_prompt("drawing_assessment", "latest")
            >>> print(template.recommended_model)
            'claude-3-haiku-20240307'
        """
        # Lazy load custom prompts if not already loaded
        if use_case not in self._prompts:
            self._lazy_load_use_case(use_case)

        if use_case not in self._prompts:
            raise PromptNotFoundError(use_case)

        versions = self._prompts[use_case]

        if version == "latest":
            version = self._get_latest_version(use_case)

        if version not in versions:
            available = list(versions.keys())
            raise PromptNotFoundError(use_case, f"{version} (available: {available})")

        template = versions[version]

        if template.deprecated:
            logger.warning(
                f"Using deprecated prompt: {use_case} v{version}. "
                "Consider upgrading to the latest version."
            )

        return template

    def render_prompt(self, template: PromptTemplate, context: Dict[str, Any]) -> str:
        """Render prompt template by substituting variables with context values.

        Uses Python string formatting to replace {variable} placeholders.
        Handles escaped braces ({{, }}) and format specifiers (e.g., {var:.2f}).

        Args:
            template: PromptTemplate to render.
            context: Dictionary mapping variable names to their values.

        Returns:
            Fully rendered prompt string ready for LLM API submission.

        Raises:
            PromptRenderError: If rendering fails (missing variables, format errors).

        Example:
            >>> template = library.get_prompt("drawing_assessment")
            >>> context = {"file_name": "DWG-001.pdf"}
            >>> prompt = library.render_prompt(template, context)
        """
        # Extract actual variables from template (handles format specs)
        actual_vars = self._extract_variables(template.template)

        # Check for missing required variables
        missing = [var for var in actual_vars if var not in context]
        if missing:
            raise PromptRenderError(
                template.name, "Missing required variables", missing
            )

        # Check for unexpected variables in context
        extra = set(context.keys()) - actual_vars
        if extra:
            logger.debug(
                f"Context contains unused variables for {template.name}: {extra}"
            )

        # Render template
        try:
            rendered = template.template.format(**context)

            # Ensure non-empty output
            if not rendered or not rendered.strip():
                raise PromptRenderError(template.name, "Rendered prompt is empty")

            return rendered.strip()

        except (KeyError, ValueError, IndexError) as e:
            raise PromptRenderError(
                template.name, f"Template formatting error: {str(e)}"
            )

    def render(
        self, use_case: str, context: Dict[str, Any], version: str = "latest"
    ) -> str:
        """Convenience method to get and render a prompt in one call.

        Args:
            use_case: Use case identifier.
            context: Dictionary with variable values.
            version: Version string or 'latest'.

        Returns:
            Rendered prompt string.

        Raises:
            PromptNotFoundError: If prompt doesn't exist.
            PromptRenderError: If rendering fails.

        Example:
            >>> prompt = library.render("drawing_assessment", {"file_name": "DWG-001.pdf"})
        """
        template = self.get_prompt(use_case, version)
        return self.render_prompt(template, context)

    def list_available_prompts(self, include_metadata: bool = False) -> Dict[str, Any]:
        """List all available prompts organized by use case.

        Args:
            include_metadata: If True, include full template metadata.

        Returns:
            Dictionary mapping use case names to version info.
            If include_metadata=False: {"use_case": ["v1.0", "v1.2"], ...}
            If include_metadata=True: {"use_case": {"v1.0": {...}, "v1.2": {...}}, ...}

        Example:
            >>> prompts = library.list_available_prompts()
            >>> print(prompts["ocr_verification"])
            ['v1.1']
        """
        # Ensure all custom prompts are loaded
        self._load_all_custom_prompts()

        if not include_metadata:
            return {
                use_case: list(versions.keys())
                for use_case, versions in self._prompts.items()
            }
        else:
            return {
                use_case: {
                    ver: {
                        "token_budget": tmpl.token_budget,
                        "recommended_model": tmpl.recommended_model,
                        "deprecated": tmpl.deprecated,
                        "variables": tmpl.variables,
                        "metadata": tmpl.metadata,
                    }
                    for ver, tmpl in versions.items()
                }
                for use_case, versions in self._prompts.items()
            }

    def add_prompt(self, prompt: PromptTemplate, overwrite: bool = False) -> None:
        """Add or update a prompt template in the library.

        Args:
            prompt: PromptTemplate to add to the library.
            overwrite: If False, raises error when overwriting existing prompt.

        Raises:
            PromptValidationError: If prompt already exists and overwrite=False.

        Example:
            >>> custom_prompt = PromptTemplate(
            ...     name="custom_check",
            ...     version="v1.0",
            ...     template="Check this: {data}",
            ...     variables=["data"],
            ...     token_budget=500,
            ...     recommended_model="gpt-3.5-turbo",
            ...     metadata={}
            ... )
            >>> library.add_prompt(custom_prompt)
        """
        if prompt.name not in self._prompts:
            self._prompts[prompt.name] = {}

        # Check for overwrite
        if not overwrite and prompt.version in self._prompts[prompt.name]:
            raise PromptValidationError(
                f"Prompt {prompt.name} v{prompt.version} already exists. "
                "Use overwrite=True to replace."
            )

        self._prompts[prompt.name][prompt.version] = prompt

        # Invalidate latest cache for this use case
        if prompt.name in self._latest_cache:
            del self._latest_cache[prompt.name]

        logger.info(f"Added prompt: {prompt.name} v{prompt.version}")

    def remove_prompt(self, use_case: str, version: str) -> None:
        """Remove a prompt from the library.

        Args:
            use_case: Use case identifier.
            version: Version to remove.

        Raises:
            PromptNotFoundError: If prompt doesn't exist.

        Example:
            >>> library.remove_prompt("custom_check", "v1.0")
        """
        if use_case not in self._prompts:
            raise PromptNotFoundError(use_case)

        if version not in self._prompts[use_case]:
            raise PromptNotFoundError(use_case, version)

        del self._prompts[use_case][version]

        # Remove use case entirely if no versions remain
        if not self._prompts[use_case]:
            del self._prompts[use_case]

        # Invalidate cache
        if use_case in self._latest_cache:
            del self._latest_cache[use_case]

        logger.info(f"Removed prompt: {use_case} v{version}")

    def deprecate_prompt(self, use_case: str, version: str) -> None:
        """Mark a prompt version as deprecated.

        Args:
            use_case: Use case identifier.
            version: Version to deprecate.

        Raises:
            PromptNotFoundError: If prompt doesn't exist.

        Example:
            >>> library.deprecate_prompt("drawing_assessment", "v1.0")
        """
        if use_case not in self._prompts:
            raise PromptNotFoundError(use_case)

        if version not in self._prompts[use_case]:
            raise PromptNotFoundError(use_case, version)

        template = self._prompts[use_case][version]

        # Create new template with deprecated flag (since frozen)
        deprecated_template = PromptTemplate(
            name=template.name,
            version=template.version,
            template=template.template,
            variables=template.variables,
            token_budget=template.token_budget,
            recommended_model=template.recommended_model,
            metadata=template.metadata,
            deprecated=True,
        )

        self._prompts[use_case][version] = deprecated_template
        logger.info(f"Deprecated prompt: {use_case} v{version}")

    def _get_latest_version(self, use_case: str) -> str:
        """Get the latest version for a use case with caching.

        Args:
            use_case: Use case identifier.

        Returns:
            Latest version string.
        """
        # Check cache first
        if use_case in self._latest_cache:
            return self._latest_cache[use_case]

        versions = self._prompts[use_case]

        # Use semantic versioning if available
        if VERSION_AVAILABLE:
            try:
                # Parse versions, handling 'v' prefix
                parsed_versions = []
                for v in versions.keys():
                    clean_v = v.lstrip("v")
                    try:
                        parsed_versions.append((version.parse(clean_v), v))
                    except Exception:
                        # Fallback to string comparison for non-semantic versions
                        parsed_versions.append((v, v))

                # Sort and get latest
                latest = max(parsed_versions, key=lambda x: x[0])[1]
            except Exception:
                # Fallback to string comparison
                latest = max(versions.keys())
        else:
            # Simple string comparison fallback
            latest = max(versions.keys())

        # Cache result
        self._latest_cache[use_case] = latest
        return latest

    def _extract_variables(self, template_str: str) -> Set[str]:
        """Extract variable names from template, handling format specs and escaped braces.

        Args:
            template_str: Template string.

        Returns:
            Set of variable names.

        Example:
            >>> vars = self._extract_variables("Hello {name}, score: {value:.2f}")
            >>> print(vars)
            {'name', 'value'}
        """
        # Pattern matches {variable} or {variable:format_spec}
        # But not {{escaped}} braces
        pattern = r"\{([a-zA-Z_]\w*)(?::[^}]*)?\}"
        variables = re.findall(pattern, template_str)
        return set(variables)

    def _validate_path(self, path_str: str) -> Path:
        """Validate and sanitize file path to prevent traversal attacks.

        Args:
            path_str: Path string to validate.

        Returns:
            Validated Path object.

        Raises:
            PromptValidationError: If path contains dangerous patterns.
        """
        # Check for dangerous patterns
        dangerous_patterns = ["..", "~", "$"]
        for pattern in dangerous_patterns:
            if pattern in path_str:
                raise PromptValidationError(f"Invalid path: contains '{pattern}'")

        try:
            path = Path(path_str).resolve()
            return path
        except Exception as e:
            raise PromptValidationError(f"Invalid path: {e}")

    def _lazy_load_use_case(self, use_case: str) -> None:
        """Lazy load custom prompts for a specific use case.

        Args:
            use_case: Use case to load.
        """
        if not self.prompts_dir.exists():
            return

        # Look for files matching use_case pattern
        patterns = [
            f"{use_case}_*.txt",
            f"{use_case}_*.yaml",
            f"{use_case}.txt",
            f"{use_case}.yaml",
        ]

        for pattern in patterns:
            for prompt_file in self.prompts_dir.glob(pattern):
                if str(prompt_file) not in self._loaded_files:
                    self._load_prompt_file(prompt_file)
                    self._loaded_files.add(str(prompt_file))

    def _load_all_custom_prompts(self) -> None:
        """Load all custom prompts from directory."""
        if not self.prompts_dir.exists():
            return

        for prompt_file in self.prompts_dir.glob("*"):
            if prompt_file.suffix in [".txt", ".yaml", ".yml"]:
                if str(prompt_file) not in self._loaded_files:
                    self._load_prompt_file(prompt_file)
                    self._loaded_files.add(str(prompt_file))

    def _load_prompt_file(self, prompt_file: Path) -> None:
        """Load a single prompt file.

        Args:
            prompt_file: Path to prompt file.
        """
        try:
            if prompt_file.suffix in [".yaml", ".yml"]:
                self._load_yaml_prompt(prompt_file)
            else:
                self._load_text_prompt(prompt_file)
        except Exception as e:
            logger.warning(f"Failed to load prompt file {prompt_file.name}: {e}")

    def _load_yaml_prompt(self, prompt_file: Path) -> None:
        """Load prompt from YAML file with full metadata.

        Expected format:
            name: use_case_name
            version: v1.0
            template: |
                Your prompt text here with {variables}
            variables:
                - var1
                - var2
            token_budget: 1000
            recommended_model: gpt-4-turbo
            metadata:
                key: value

        Args:
            prompt_file: Path to YAML file.
        """
        if not YAML_AVAILABLE:
            logger.warning(
                "YAML support not available. Install PyYAML to load .yaml files."
            )
            return

        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Validate required fields
            required = ["name", "version", "template"]
            missing = [f for f in required if f not in data]
            if missing:
                raise PromptValidationError(
                    f"YAML file missing required fields: {missing}"
                )

            # Extract or infer variables
            if "variables" not in data:
                data["variables"] = list(self._extract_variables(data["template"]))

            # Use defaults for optional fields
            prompt = PromptTemplate(
                name=data["name"],
                version=data["version"],
                template=data["template"],
                variables=data["variables"],
                token_budget=data.get("token_budget", self.default_token_budget),
                recommended_model=data.get("recommended_model", self.default_model),
                metadata=data.get("metadata", {}),
                deprecated=data.get("deprecated", False),
            )

            self.add_prompt(prompt, overwrite=True)

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {prompt_file.name}: {e}")
        except Exception as e:
            logger.error(f"Error loading YAML prompt {prompt_file.name}: {e}")

    def _load_text_prompt(self, prompt_file: Path) -> None:
        """Load prompt from plain text file.

        File naming: {use_case}_{version}.txt or {use_case}.txt (defaults to v1.0)

        Args:
            prompt_file: Path to text file.
        """
        try:
            stem = prompt_file.stem

            # Parse filename for name and version
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].startswith("v"):
                name, version = parts
            else:
                name = stem
                version = "v1.0"

            # Read template
            template_text = prompt_file.read_text(encoding="utf-8")

            if not template_text.strip():
                logger.warning(f"Skipping empty prompt file: {prompt_file.name}")
                return

            # Extract variables
            variables = list(self._extract_variables(template_text))

            prompt = PromptTemplate(
                name=name,
                version=version,
                template=template_text,
                variables=variables,
                token_budget=self.default_token_budget,
                recommended_model=self.default_model,
                metadata={"source": "file", "filename": prompt_file.name},
            )

            self.add_prompt(prompt, overwrite=True)

        except Exception as e:
            logger.error(f"Error loading text prompt {prompt_file.name}: {e}")

    def _load_builtin_prompts(self) -> None:
        """Load built-in prompt templates for core use cases.

        Hardcodes prompt templates for the four primary use cases:
        - drawing_assessment: Quality and complexity evaluation
        - ocr_verification: OCR error correction
        - entity_extraction: Structured data extraction
        - shape_validation: Detection validation

        This method is called during __init__ and should not be called directly.
        """
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
