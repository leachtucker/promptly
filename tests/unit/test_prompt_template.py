"""
Tests for PromptTemplate and PromptMetadata
"""

import pytest
from datetime import datetime
from promptly.core.templates import PromptTemplate, PromptMetadata


class TestPromptMetadata:
    """Test PromptMetadata functionality"""

    def test_prompt_metadata_creation(self):
        """Test basic PromptMetadata creation"""
        metadata = PromptMetadata(
            name="test_prompt", description="A test prompt", tags=["test", "example"]
        )

        assert metadata.name == "test_prompt"
        assert metadata.description == "A test prompt"
        assert metadata.tags == ["test", "example"]
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.created_at, datetime)

    def test_prompt_metadata_defaults(self):
        """Test PromptMetadata with defaults"""
        metadata = PromptMetadata(name="test")

        assert metadata.name == "test"
        assert metadata.description == ""
        assert metadata.tags == []
        assert metadata.version == "1.0.0"
        assert isinstance(metadata.created_at, datetime)


class TestPromptTemplate:
    """Test PromptTemplate functionality"""

    def test_prompt_template_creation(self):
        """Test basic PromptTemplate creation"""
        template = PromptTemplate(template="Hello {{ name }}!", name="greeting")

        assert template.template == "Hello {{ name }}!"
        assert template.name == "greeting"
        assert isinstance(template.metadata, PromptMetadata)

    def test_prompt_template_render(self):
        """Test template rendering with variables"""
        template = PromptTemplate(
            template="Hello {{ name }}, you are {{ age }} years old.", name="greeting"
        )

        result = template.render(name="Alice", age=25)
        assert result == "Hello Alice, you are 25 years old."

    def test_prompt_template_render_with_defaults(self):
        """Test template rendering with default values"""
        template = PromptTemplate(
            template="Hello {{ name|default('World') }}!", name="greeting"
        )

        result = template.render()
        assert result == "Hello World!"

    def test_prompt_template_render_missing_variable(self):
        """Test template rendering with missing required variable"""
        template = PromptTemplate(template="Hello {{ name }}!", name="greeting")

        with pytest.raises(Exception):  # Jinja2 will raise UndefinedError
            template.render()

    def test_prompt_template_render_with_env_vars(self):
        """Test template rendering with environment variables"""
        template = PromptTemplate(
            template="Hello {{ name }}, environment: {{ ENV_NAME }}",
            name="greeting",
            env_vars={"ENV_NAME": "test"},
        )

        result = template.render(name="Alice")
        assert result == "Hello Alice, environment: test"

    def test_prompt_template_auto_name(self):
        """Test template with auto-generated name"""
        template = PromptTemplate(template="Hello {{ name }}!")

        assert template.name.startswith("prompt_")
        assert template.metadata.name == template.name

    def test_prompt_template_compilation_caching(self):
        """Test that template compilation is cached"""
        template = PromptTemplate(template="Hello {{ name }}!", name="greeting")

        # First render should compile template
        result1 = template.render(name="Alice")
        assert result1 == "Hello Alice!"

        # Second render should use cached compilation
        result2 = template.render(name="Bob")
        assert result2 == "Hello Bob!"

        # Verify _compiled_template is set
        assert template._compiled_template is not None

    def test_prompt_template_validation(self):
        """Test template validation methods"""
        template = PromptTemplate(
            template="Hello {{ name }}, you are {{ age }} years old.", name="greeting"
        )

        # Test variables
        variables = template.get_variables()
        assert "name" in variables
        assert "age" in variables

        # Test that variables are correctly extracted
        assert len(variables) == 2  # name and age
        
        # Test validate_variables
        assert template.validate_variables({"name": "Alice", "age": 25})
        assert not template.validate_variables({"name": "Alice"})  # Missing age
        assert template.validate_variables(
            {"name": "Alice", "age": 25, "extra": "value"}
        )  # Extra variable is OK
        
        # Test get_validation_errors
        errors = template.get_validation_errors({"name": "Alice"})
        assert len(errors) == 1
        assert "Missing required variables: age" in errors[0]
        
        errors = template.get_validation_errors({"name": "Alice", "age": 25})
        assert len(errors) == 0
