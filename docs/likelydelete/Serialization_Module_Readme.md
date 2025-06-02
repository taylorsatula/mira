# JSON Serialization Module

This document describes the standardized JSON serialization utility module for the project.

## Overview

The `serialization.py` module provides consistent JSON serialization and deserialization functionality across the codebase. It handles common types like `datetime` and `UUID` objects, and supports custom objects that implement `to_dict()` and `from_dict()` methods.

## Usage

### Basic Usage

```python
from serialization import to_json, from_json

# Serialize an object to JSON
json_data = to_json(my_object)

# Deserialize JSON to a dictionary
data = from_json(json_data)
```

### Serializing Custom Objects

For a class to be serializable:

1. Implement a `to_dict()` method that returns a dictionary representation of the object.
2. Optionally implement a class method `from_dict(cls, data)` that creates an instance from a dictionary.

```python
class MyClass:
    def to_dict(self):
        return {
            "field1": self.field1,
            "field2": self.field2
        }
    
    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.field1 = data.get("field1")
        obj.field2 = data.get("field2")
        return obj
```

### Deserializing to Custom Objects

When deserializing, you can provide a class to deserialize to:

```python
from serialization import from_json

# Deserialize JSON to a MyClass instance
obj = from_json(json_data, MyClass)
```

### Customization

The `to_json` function accepts all the same parameters as `json.dumps()`, allowing for customization:

```python
# Serialize with custom indentation
json_data = to_json(my_object, indent=4)

# Serialize with sorted keys
json_data = to_json(my_object, sort_keys=True)
```

## Supported Types

The serialization module automatically handles:

- Basic Python types (`str`, `int`, `float`, `bool`, `list`, `dict`)
- `datetime.datetime` objects (serialized as ISO format strings)
- `datetime.date` objects (serialized as ISO format strings)
- `uuid.UUID` objects (serialized as strings)
- Objects with a `to_dict()` method

## Benefits

1. **Consistency**: Uniform serialization behavior across the codebase
2. **Maintainability**: Single location for serialization code
3. **Extensibility**: Easy to add support for new types
4. **Error Handling**: Consistent error messages and handling
5. **Type Safety**: Better type checking and validation

## Implementation Details

The module consists of:

- `to_json(obj, indent=2, **kwargs)`: Serializes objects to JSON strings
- `from_json(json_str, cls=None)`: Deserializes JSON strings to objects
- `_json_serializer(obj)`: Internal function for handling non-JSON-serializable objects

The `_json_serializer` function is used by `to_json` as the `default` parameter for `json.dumps()`. It provides custom handling for the supported types.