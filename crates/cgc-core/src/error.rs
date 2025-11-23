//! Error types for the core graph engine.

use thiserror::Error;

/// Core graph engine errors
#[derive(Error, Debug)]
pub enum GraphError {
    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    /// Edge not found
    #[error("Edge not found: {0}")]
    EdgeNotFound(String),
    
    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

/// Result type for graph operations
pub type Result<T> = std::result::Result<T, GraphError>;
