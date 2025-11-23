//! Core graph data structures: Node, Edge, and CausalGraph.
//!
//! Implements: Requirements 4, 5, 7

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for nodes
pub type NodeId = Uuid;

/// Unique identifier for edges
pub type EdgeId = Uuid;

/// Unique identifier for requirements
pub type RequirementId = String;

/// A node representing a semantic region in embedding space
///
/// Implements: Requirements 4, 5, 7
#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier
    pub id: NodeId,
    
    /// Collection of all embedding vectors mapped to this node
    pub embeddings: Vec<Vec<f32>>,
    
    /// Centroid (mean) of all embeddings
    pub centroid: Vec<f32>,
    
    /// Variance of the embedding cluster
    pub variance: f32,
    
    /// Natural language labels
    pub labels: Vec<String>,
    
    /// Confidence in the node identity
    pub confidence: f32,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Traceability
    pub source_requirements: Vec<RequirementId>,
}

/// Edge relationship types
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum RelationType {
    /// P→Q: from (premise) implies to (conclusion)
    Implication,
    
    /// from (constraint) constrains to (constrained)
    ConstraintOn,
    
    /// from (assumption) is assumption for to (dependent)
    AssumptionFor,
    
    /// from (context) contextualizes to (contextualized)
    ContextualizedBy,
    
    /// from (negating) negates to (negated)
    Negation,
}

/// An edge encoding a causal relationship
///
/// Implements: Requirements 4, 7
#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier
    pub id: EdgeId,
    
    /// Source node
    pub from: NodeId,
    
    /// Target node
    pub to: NodeId,
    
    /// Type of relationship
    pub relation_type: RelationType,
    
    /// Confidence in this relationship
    pub confidence: f32,
    
    /// Timestamp
    pub created_at: DateTime<Utc>,
    
    /// Traceability
    pub source_requirements: Vec<RequirementId>,
}

/// Causal statement from parser (text only, no embeddings yet)
///
/// Implements: Requirements 1, 2
#[derive(Clone, Serialize, Deserialize)]
pub struct CausalStatement {
    /// Unique identifier
    pub id: String,
    
    /// Premise text
    pub premise_text: String,
    
    /// Conclusion text
    pub conclusion_text: String,
    
    /// Relation type
    pub relation_type: RelationType,
    
    /// Confidence score
    pub confidence: f32,
    
    /// Source requirement
    pub source_requirement: RequirementId,
}
