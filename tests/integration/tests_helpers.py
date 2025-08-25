"""
Helper functions for testing. If complex logic is needed, add the function here in this file.
"""

def validate_product_list(response, expected_length=None, required_fields=None):
    """
    Validate that response is a list of product dictionaries with required fields

    Args:
        response: The HTTP response object
        expected_length: Expected number of items
        required_fields: List of required fields in each item (optional)

    Raises:
        AssertionError: If validation fails
    """
    data = response.json()

    # Check if response is a list
    assert isinstance(data, list), f"Response is not a list, got: {type(data)}"
    if expected_length:
        assert len(data) == expected_length, f"Expected {expected_length} items, got {len(data)}"
    assert all(isinstance(item, dict) for item in data), "All items must be dictionaries"

    # Fix the syntax error - check required fields if provided
    if required_fields:
        for item in data:
            assert all(field in item for field in required_fields), f"Item missing required fields: {item}"

    print(f"✅ Validated list of {len(data)} product items with fields: {required_fields}")
