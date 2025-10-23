import json
from datetime import datetime
from typing import Any, Optional

import jinja2
from pydantic import BaseModel, Field


class PromptMetadata(BaseModel):
    """Metadata for prompt templates"""

    name: str
    version: str = "1.0.0"
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)


class PromptTemplate:
    """Core prompt template with variable interpolation"""

    def __init__(
        self,
        template: str,
        name: Optional[str] = None,
        metadata: Optional[PromptMetadata] = None,
        env_vars: Optional[dict[str, Any]] = None,
    ):
        self.template = template
        self.name = name or f"prompt_{id(self)}"
        self.metadata = metadata or PromptMetadata(name=self.name)
        self.env_vars = env_vars or {}

        # Set up Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined
        )
        self._compiled_template: Optional[jinja2.Template] = None

    def render(self, **kwargs: Any) -> str:
        """Render the template with given variables"""
        if self._compiled_template is None:
            self._compiled_template = self.jinja_env.from_string(self.template)

        # At this point, _compiled_template is guaranteed to be a Template object
        assert self._compiled_template is not None

        # Merge env vars and kwargs (kwargs take precedence)
        context = {**self.env_vars, **kwargs}

        try:
            return self._compiled_template.render(**context)
        except jinja2.UndefinedError as e:
            raise ValueError(f"Missing template variable: {e}") from e

    def get_variables(self) -> list[str]:
        """Extract all variables from the template"""
        if self._compiled_template is None:
            self._compiled_template = self.jinja_env.from_string(self.template)

        # At this point, _compiled_template is guaranteed to be a Template object
        assert self._compiled_template is not None
        return [
            node.name
            for node in self._compiled_template.environment.parse(self.template).find_all(
                jinja2.nodes.Name  # pyright: ignore[reportAttributeAccessIssue]
            )
        ]

    def validate_variables(self, variables: dict[str, Any]) -> bool:
        """Validate that all required template variables are provided

        Args:
            variables: Dictionary of variables to validate

        Returns:
            True if all required variables are provided, False otherwise
        """
        required_vars = set(self.get_variables())
        provided_vars = set(variables.keys())

        # Check if all required variables are provided
        missing_vars = required_vars - provided_vars
        if missing_vars:
            return False

        return True

    def get_validation_errors(self, variables: dict[str, Any]) -> list[str]:
        """Get detailed validation errors for template variables

        Args:
            variables: Dictionary of variables to validate

        Returns:
            List of error messages describing validation issues
        """
        errors = []
        required_vars = set(self.get_variables())
        provided_vars = set(variables.keys())

        # Check for missing variables
        missing_vars = required_vars - provided_vars
        if missing_vars:
            errors.append(f"Missing required variables: {', '.join(sorted(missing_vars))}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "template": self.template,
            "name": self.name,
            "metadata": self.metadata.model_dump(),
            "env_vars": self.env_vars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary"""
        metadata = PromptMetadata(**data.get("metadata", {}))
        return cls(
            template=data["template"],
            name=data.get("name", ""),
            metadata=metadata,
            env_vars=data.get("env_vars", {}),
        )

    def save(self, filepath: str) -> None:
        """Save template to file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "PromptTemplate":
        """Load template from file"""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)
