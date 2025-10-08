from pydantic import BaseModel
from typing import get_origin


def simple_schema(model: type[BaseModel]) -> dict[str, str]:
    """
    Generate a simple JSON schema from a Pydantic model.
    
    Args:
        model: A Pydantic BaseModel class
        
    Returns:
        Dictionary with format { key: type }
    """
    schema = {}
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        
        # Check if it's a generic type (List[str], Dict[str, int], etc.)
        if get_origin(annotation) is not None:
            # It's a generic type, use string representation to preserve inner types
            type_str = str(annotation).replace('typing.', '')
        elif hasattr(annotation, '__name__'):
            # Simple type like str, int, float
            type_str = annotation.__name__  # type: ignore
        else:
            # Fallback for other complex types
            type_str = str(annotation).replace('typing.', '')
        
        schema[field_name] = type_str
    
    return schema