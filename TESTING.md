# Testing Guide

This document describes the Test-Driven Development (TDD) workflow and testing infrastructure for the Causal Graph Compiler.

## TDD Workflow

We follow strict TDD with the Red-Green-Refactor cycle:

### 1. Red Phase: Write Failing Tests
- Write tests **before** implementation
- Tests should fail initially (compilation errors or assertion failures)
- Verify the test fails for the right reason

### 2. Green Phase: Make Tests Pass
- Write minimal code to make the test pass
- Don't worry about perfection yet
- Focus on making the test green

### 3. Refactor Phase: Improve Code
- Refactor while keeping tests green
- Improve code quality, readability, performance
- Tests provide safety net for refactoring

## Test Hierarchy

Tests are organized in three layers:

```
┌─────────────────────────────────────┐
│   Integration Tests (42)            │  ← One per requirement
│   Verify complete user stories      │
├─────────────────────────────────────┤
│   Unit Tests (210+)                 │  ← One per acceptance criterion
│   Verify specific functionality     │
├─────────────────────────────────────┤
│   Doctests (100+)                   │  ← Multiple per public function
│   Verify usage examples             │
└─────────────────────────────────────┘
```

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Verify complete requirements end-to-end
- **Naming**: `test_requirement_N_description`
- **Example**: `test_requirement_4_graph_construction`
- **One test per requirement user story**

### Unit Tests
- **Location**: Within each crate's `src/` directory (e.g., `#[cfg(test)] mod tests`)
- **Purpose**: Verify specific acceptance criteria
- **Naming**: `test_reqN_acM_description`
- **Example**: `test_req5_ac4_similarity_threshold`
- **One test per acceptance criterion**

### Property-Based Tests
- **Location**: `tests/property/`
- **Purpose**: Verify universal properties across all inputs
- **Naming**: `prop_property_N_description`
- **Example**: `prop_property_1_centroid_based_node_identity`
- **Minimum 100 iterations per property**

### Doctests
- **Location**: Embedded in documentation comments (`///` or `//!`)
- **Purpose**: Verify usage examples and keep docs accurate
- **All public functions must have at least one doctest**

## Test Naming Conventions

### Integration Tests
```rust
/// Integration test for Requirement 4: Graph Construction
/// Validates: Requirement 4
#[test]
fn test_requirement_4_graph_construction() {
    // Test complete user story
}
```

### Unit Tests
```rust
/// Unit test for Requirement 5, AC 5.4
/// Validates: Requirement 5, AC 5.4
#[test]
fn test_req5_ac4_similarity_threshold() {
    // Test specific acceptance criterion
}
```

### Property Tests
```rust
/// Property 1: Centroid-Based Node Identity
/// Validates: Requirements 4.5, 5.4, 7.2
#[test]
fn prop_property_1_centroid_based_node_identity() {
    // Property-based test with 100+ iterations
}
```

### Doctests
```rust
/// Add a node to the graph.
///
/// Implements: Requirement 4, AC 4.3
///
/// # Examples
///
/// ```
/// use cgc_core::graph::Node;
///
/// let node = Node::new("temperature > 100°C");
/// assert!(node.labels.len() > 0);
/// ```
pub fn add_node(&mut self, text: String) -> Result<NodeId> {
    // Implementation
}
```

## Property-Based Testing

We use `proptest` for property-based testing with minimum 100 iterations.

### Configuration

Add to `Cargo.toml`:
```toml
[dev-dependencies]
proptest = { workspace = true }
```

### Example Property Test

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    /// Property 13: Serialization Round-Trip
    /// Validates: Requirements 17.1, 17.2, 17.3
    #[test]
    fn prop_serialization_round_trip(
        num_nodes in 1..100usize,
        num_edges in 1..200usize,
    ) {
        let graph = generate_random_graph(num_nodes, num_edges);
        
        // Serialize
        let json = serde_json::to_string(&graph).unwrap();
        
        // Deserialize
        let deserialized: CausalGraph = serde_json::from_str(&json).unwrap();
        
        // Should be equivalent
        assert_graphs_equivalent(&graph, &deserialized);
    }
}
```

## Test Coverage

We use `tarpaulin` for code coverage reporting.

### Installation
```bash
cargo install cargo-tarpaulin
```

### Running Coverage
```bash
# Generate HTML report
cargo tarpaulin --out Html

# Generate multiple formats
cargo tarpaulin --out Html --out Xml --out Lcov
```

### Coverage Thresholds
- **Overall**: ≥80%
- **Critical paths** (parsing, graph ops, consistency): ≥90%
- **Error handling**: ≥85%

### Configuration

Create `.tarpaulin.toml`:
```toml
[report]
out = ["Html", "Xml", "Lcov"]

[coverage]
exclude-files = [
    "*/tests/*",
    "*/examples/*",
]

[thresholds]
line = 80
branch = 75
```

## Test Dependency Structure

Integration tests depend on unit tests:

```rust
// Integration test
#[test]
fn test_requirement_4_graph_construction() {
    // Verify all unit tests pass first
    assert!(unit_tests_pass_for_requirement_4());
    
    // Then run integration test
    // ...
}
```

## Running Tests

### All Tests
```bash
cargo test
```

### Specific Crate
```bash
cargo test -p cgc-core
```

### Integration Tests Only
```bash
cargo test --test '*'
```

### Property Tests Only
```bash
cargo test --test property
```

### With Coverage
```bash
cargo tarpaulin --out Html
```

### Verbose Output
```bash
cargo test -- --nocapture
```

## Requirement Completion Criteria

A requirement is considered complete when:

1. ✅ All acceptance criteria have passing unit tests
2. ✅ Integration test for the requirement passes
3. ✅ All public functions have doctests
4. ✅ Property tests (if applicable) pass with 100+ iterations
5. ✅ Code coverage meets thresholds (≥80% overall, ≥90% critical paths)
6. ✅ All tests pass in CI/CD pipeline

## Best Practices

### DO:
- ✅ Write tests before implementation
- ✅ Keep tests simple and focused
- ✅ Use descriptive test names
- ✅ Test edge cases and error conditions
- ✅ Run tests frequently during development
- ✅ Keep tests fast (< 1s per test when possible)

### DON'T:
- ❌ Skip the red phase (always see tests fail first)
- ❌ Write tests after implementation
- ❌ Test implementation details (test behavior, not internals)
- ❌ Use mocks when real implementations are simple
- ❌ Ignore failing tests
- ❌ Commit code with failing tests

## Example Test Structure

```rust
// crates/cgc-core/src/graph.rs

/// Node in the causal graph
pub struct Node {
    // ...
}

impl Node {
    /// Create a new node
    ///
    /// Implements: Requirement 4, AC 4.3
    ///
    /// # Examples
    ///
    /// ```
    /// use cgc_core::graph::Node;
    ///
    /// let node = Node::new("temperature > 100°C");
    /// assert_eq!(node.labels.len(), 1);
    /// ```
    pub fn new(label: String) -> Self {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit test for Requirement 4, AC 4.3
    /// Validates: Requirement 4, AC 4.3
    #[test]
    fn test_req4_ac3_node_creation() {
        let node = Node::new("test".to_string());
        assert_eq!(node.labels.len(), 1);
    }
}
```

## Continuous Integration

Tests run automatically on:
- Every commit
- Every pull request
- Before merging to main

CI pipeline fails if:
- Any test fails
- Coverage drops below threshold
- Doctests fail

## Resources

- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Proptest Documentation](https://docs.rs/proptest/)
- [Tarpaulin Documentation](https://github.com/xd009642/tarpaulin)

---

**Remember**: Tests are not just verification - they're documentation, design tools, and safety nets. Write them first, keep them green, and let them guide your implementation.
