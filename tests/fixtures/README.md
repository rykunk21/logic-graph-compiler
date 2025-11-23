# Test Fixtures

This directory contains test data and fixtures used across integration and unit tests.

## Structure

- `requirements/` - Sample requirement files (CSV, JSON)
- `graphs/` - Sample graph files for testing
- `expected/` - Expected output files for comparison

## Usage

```rust
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn test_with_fixture() {
    let requirements = fixture_path("requirements/sample.csv");
    // Use fixture in test
}
```

## Guidelines

- Keep fixtures small and focused
- Use realistic data that matches actual use cases
- Document what each fixture tests
- Version control all fixtures
