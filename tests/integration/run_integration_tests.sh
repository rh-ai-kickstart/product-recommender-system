#!/bin/bash


# Function to validate URL
validate_url() {
    local url=$1
    local name=$2

    # Check if URL is empty or contains 'null'
    if [ -z "$url" ] || [[ "$url" == *"null"* ]]; then
        echo "❌ $name URL is invalid: $url"
        return 1
    fi
    echo "✅ $name URL is valid"
    return 0
}

# Get URLs dynamically
export TEST_FRONTEND_URL=$(oc get routes product-recommender-system-frontend -o json | jq -r '"https://" + .spec.host')
export TEST_FEAST_URL=$(oc get routes feast-feast-recommendation-ui -o json | jq -r '"https://" + .spec.host')

# Generate unique timestamp for test emails
export TEST_TIMESTAMP=$(date +%s)

echo "Testing with:"
echo "Frontend URL: $TEST_FRONTEND_URL"
echo "Feast URL: $TEST_FEAST_URL"
echo ""

# Validate critical URLs (Frontend and Backend)
echo "Validating URLs..."
validate_url "$TEST_FRONTEND_URL" "Frontend"
frontend_valid=$?

validate_url "$TEST_FEAST_URL" "Feast"
feast_valid=$?

# Exit if either critical URL is invalid
if [ $frontend_valid -ne 0 ] || [ $feast_valid -ne 0 ]; then
    echo "ERROR: One or more critical URLs are invalid. Exiting."
    exit 1
fi

echo "All critical URLs validated successfully. Starting tests..."
echo ""

# Determine test target
if [ -n "$1" ]; then
    # Specific test file provided
    TEST_TARGET="$1"
    if [ ! -f "$TEST_TARGET" ]; then
        echo "❌ Test file not found: $TEST_TARGET"
        exit 1
    fi
    echo "Running specific test: $TEST_TARGET"
else
    # Run all tests in current directory
    TEST_TARGET="."
    echo "Running all tests in current directory"
fi

echo ""

# Run pytest with the determined target
PYTHONPATH=. pytest "$TEST_TARGET" -v
