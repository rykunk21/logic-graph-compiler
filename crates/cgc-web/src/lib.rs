//! # Causal Graph Compiler - Web Server
//!
//! Web interface and API for the Causal Graph Compiler.
//!
//! This crate provides:
//! - HTTP API endpoints
//! - Static file serving
//! - WebSocket support for real-time updates
//!
//! Implements: Requirements 26, 33

#![warn(missing_docs)]

/// API routes
pub mod routes;

/// WebSocket handlers
pub mod ws;

/// Static file serving
pub mod static_files;
