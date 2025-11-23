//! # Causal Graph Compiler - Parser
//!
//! Requirements parser for extracting causal statements from natural language.
//!
//! This crate provides:
//! - Temporal keyword detection
//! - Causal statement extraction
//! - Constraint and assumption extraction
//! - Logical decomposition
//!
//! Implements: Requirements 1, 2, 19

#![warn(missing_docs)]

/// Temporal keyword detection
pub mod keywords;

/// Causal statement extraction
pub mod extractor;

/// Logical decomposition
pub mod decomposition;
