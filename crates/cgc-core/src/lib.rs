//! # Causal Graph Compiler - Core
//!
//! Core data structures and graph engine for the Causal Graph Compiler.
//!
//! This crate provides:
//! - Node and Edge data structures
//! - CSR-based CausalGraph implementation
//! - Embedding index with HNSW
//! - Consistency checking
//! - Semantic refinement operations
//!
//! Implements: Requirements 4, 5, 6, 7, 19

#![warn(missing_docs)]

/// Core error types
pub mod error;

/// Node and edge data structures
pub mod graph;

/// Embedding index for similarity search
pub mod embedding;

/// Consistency checking
pub mod consistency;

/// Semantic refinement (node splitting and aggregation)
pub mod refinement;
