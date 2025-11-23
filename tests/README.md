# Integration Tests

This directory contains integration tests that verify complete requirements end-to-end.

## Structure

- `integration/` - Integration tests (one per requirement)
- `property/` - Property-based tests for correctness properties
- `fixtures/` - Test data and fixtures

## Running Tests

```bash
# Run all tests
cargo test

# Run integration tests only
cargo test --test '*'

# Run property tests only
cargo test --test property

# Run with coverage
cargo tarpaulin --out Html
```

## Test Organization

- Integration tests: `test_requirement_N_description`
- Unit tests: `test_reqN_acM_description`
- Property tests: `prop_property_N_description`

Implements: Requirements 35, 36, 37, 38, 39, 42
