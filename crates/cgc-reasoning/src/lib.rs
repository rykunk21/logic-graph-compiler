//! # Causal Graph Compiler - Reasoning Engine
//!
//! Logical inference and test generation from causal graphs.
//!
//! This crate provides:
//! - Modus ponens inference
//! - Test case generation
//! - Forward and backward chaining
//! - Query processing
//!
//! Implements: Requirements 8, 10, 12, 16, 19

#![warn(missing_docs)]

/// Modus ponens and inference rules
pub mod inference;

/// Test case generation
pub mod test_gen;

/// Query processing
pub mod query;

/// Design-space exploration
pub mod exploration;
