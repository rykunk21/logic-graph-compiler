# Implementation Plan

## Phase 1: Project Foundation and Development Process

- [x] 1. Create comprehensive README.md
  - Document project purpose and architecture overview
  - Add installation instructions for all platforms (Linux, Windows, macOS)
  - Include quick start guide with examples
  - Document CLI commands and usage
  - Add contribution guidelines and development setup
  - Include links to requirements and design documents
  - _Requirements: 24.1, 24.2, 24.3, 41.1_

- [ ] 2. Set up monorepo structure with Cargo workspace
  - Create root Cargo.toml with workspace configuration
  - Set up directory structure: crates/, services/, proto/, tests/, installer/
  - Configure shared dependencies at workspace level
  - Add .gitignore for Rust and Docker artifacts
  - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_

- [ ] 3. Establish TDD workflow and test infrastructure
  - Document TDD process: write tests first (red), implement (green), refactor
  - Set up test organization: tests/ for integration, unit tests in crates, property tests
  - Configure proptest with minimum 100 iterations per property test
  - Set up tarpaulin for coverage reporting (≥80% overall, ≥90% critical paths)
  - Implement test naming conventions: test_requirement_N, test_reqN_acM, prop_property_N
  - Create test dependency structure: integration → unit → doctests
  - Document requirement completion criteria: all AC tests pass, coverage meets threshold
  - _Requirements: 35.1, 35.2, 35.3, 35.4, 35.5, 36.2, 36.5, 37.2, 37.5, 38.1, 38.2, 38.3, 38.4, 38.5, 42.1, 42.2, 42.3, 42.4, 42.5_

- [ ] 4. Define Protocol Buffer schemas for inter-service communication
  - Create proto/embedding_service.proto with Embed, BatchEmbed, FindSimilar RPCs
  - Create proto/nlp_service.proto for NLP operations
  - Define message types with proper field types and validation
  - Document schema contracts and type safety guarantees
  - _Requirements: 20.3, 20.4, 20.5_

## Phase 2: Core Data Models

- [ ] 5. Implement core data models in cgc-core crate
- [ ] 5.1 Create Node struct with centroid-based semantic region
  - Implement Node with id, embeddings (Vec<Vec<f32>>), centroid, variance, labels, confidence
  - Add timestamps, source_requirements for traceability
  - Add serde serialization/deserialization support
  - Write doctests demonstrating Node creation and centroid calculation
  - _Requirements: 4.3, 4.6, 4.7, 5.1, 5.6, 5.7, 5.10, 5.11, 39.1_

- [ ] 5.2 Create Edge struct with relationship types
  - Implement Edge with id, from, to, relation_type, confidence, timestamps, source_requirements
  - Define RelationType enum (Implication, ConstraintOn, AssumptionFor, ContextualizedBy, Negation)
  - Write doctests for Edge creation and RelationType usage
  - _Requirements: 4.2, 4.7, 39.1_


- [ ] 5.3 Create CausalStatement struct for parsed requirements
  - Implement CausalStatement with premise_text, conclusion_text, relation_type, confidence
  - Add traceability to source requirements
  - Write doctests for CausalStatement usage
  - _Requirements: 1.1, 1.2, 1.5, 2.4, 39.1_

- [ ]* 5.4 Write property test for Node serialization round-trip
  - **Property 16: Serialization Round-Trip**
  - **Validates: Requirements 17.1, 17.2, 17.3**

- [ ]* 5.5 Write property test for embedding collection preservation
  - **Property 17: Embedding Collection Preservation**
  - **Validates: Requirements 17.5**

- [ ]* 5.6 Write property test for centroid convergence
  - **Property 3: Centroid Convergence**
  - **Validates: Requirements 5.10**

- [ ]* 5.7 Write property test for variance consistency
  - **Property 4: Variance Consistency**
  - **Validates: Requirements 5.11**

## Phase 3: Graph Engine with CSR

- [ ] 6. Implement CSR-based CausalGraph structure
- [ ] 6.1 Create CausalGraph with CSR graph backend
  - Use csr crate for compressed sparse row graph
  - Implement node_id_to_index and index_to_node_id mappings
  - Add traceability map for requirements tracking
  - Write doctests for CausalGraph initialization
  - _Requirements: 4.1, 7.2, 19.1, 39.1_

- [ ] 6.2 Implement centroid calculation helper functions
  - Implement compute_centroid() to calculate mean of embedding vectors
  - Implement compute_variance() to measure cluster tightness
  - Implement compute_confidence() based on variance and cluster size
  - Implement euclidean_distance() for variance calculation
  - Write doctests for all helper functions
  - _Requirements: 5.10, 5.11, 39.1_

- [ ] 6.3 Implement add_node with centroid-based deduplication and variance checking
  - Compare new embedding against existing node centroids using threshold (0.85)
  - Check if merge would exceed max variance threshold (0.8) - reject if so
  - If merging: add embedding to collection, recalculate centroid, update variance
  - Flag nodes with variance > 0.5 for review
  - Update HNSW index with new centroid position
  - Create new node if no similar centroid found or variance would exceed threshold
  - Write doctests showing node addition, merging, and variance rejection
  - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 5.4, 5.6, 5.9, 5.10, 5.11, 5.12, 5.13, 7.2, 9.3, 9.4, 39.1, 39.2_

- [ ] 6.4 Implement add_edge for creating causal relations
  - Add directed edge with relation type annotation
  - Update CSR graph structure
  - Maintain traceability to source requirements
  - Support multiple edges from same node for different contexts
  - Write doctests for edge creation
  - _Requirements: 4.2, 4.7, 4.8, 4.9, 7.1, 7.4, 39.1_

- [ ] 6.5 Implement node_edges for querying node relationships
  - Return both incoming and outgoing edges for a node
  - Support filtering by relation type
  - Enable role fluidity queries
  - Write doctests for querying node edges
  - _Requirements: 7.3, 7.5, 7.6, 39.1_

- [ ] 6.6 Implement traverse for causal chain traversal
  - DFS traversal following specific relation types
  - Support max depth limiting
  - Return all paths from start node
  - Write doctests for graph traversal
  - _Requirements: 7.4, 7.5, 8.2, 39.1_

- [ ]* 6.7 Write property test for centroid-based node identity
  - **Property 1: Centroid-Based Node Identity**
  - **Validates: Requirements 4.5, 5.4, 7.2**

- [ ]* 6.8 Write property test for node label and embedding accumulation
  - **Property 2: Node Label and Embedding Accumulation**
  - **Validates: Requirements 5.6, 5.7, 5.8**

- [ ]* 6.9 Write property test for edge directionality
  - **Property 5: Edge Directionality**
  - **Validates: Requirements 4.2**

- [ ]* 6.10 Write property test for role fluidity
  - **Property 6: Role Fluidity**
  - **Validates: Requirements 7.1, 7.3, 7.4**

- [ ]* 6.11 Write property test for multiple edges from same node
  - **Property 7: Multiple Edges from Same Node**
  - **Validates: Requirements 4.9, 7.4, 9.4**

- [ ]* 6.12 Write property test for transitive traversal
  - **Property 8: Transitive Traversal**
  - **Validates: Requirements 7.4**

- [ ] 7. Implement EmbeddingIndex with HNSW
- [ ] 7.1 Create EmbeddingIndex wrapper around HNSW
  - Initialize HNSW index with cosine similarity metric
  - Support configurable similarity threshold
  - Implement add() to add node centroid to index
  - Implement find_similar() to search by centroid
  - Implement update() to update centroid position when node changes
  - Implement batch operations for efficiency
  - Write doctests for embedding index operations
  - _Requirements: 5.3, 5.4, 5.5, 5.9, 32.5, 39.1_

- [ ] 7.2 Integrate EmbeddingIndex into CausalGraph
  - Use index in add_node for centroid similarity checks
  - Update index when centroid changes after adding embeddings
  - Rebuild index on deserialization
  - Support re-embedding when model updates
  - _Requirements: 4.4, 4.5, 5.8, 5.9, 5.14, 17.4_


## Phase 4: Python Embedding Service with gRPC

- [ ] 8. Implement Python embedding service
- [ ] 8.1 Create embedding service with sentence-transformers
  - Load all-MiniLM-L6-v2 model
  - Implement Embed RPC for single text embedding
  - Implement BatchEmbed RPC for batch processing
  - Return 384-dimensional embedding vectors
  - _Requirements: 5.1, 5.2, 20.1_

- [ ] 8.2 Add HNSW index to embedding service
  - Initialize hnswlib index for similarity search
  - Implement FindSimilar RPC with cosine similarity
  - Support configurable similarity threshold
  - _Requirements: 5.5, 32.5_

- [ ] 8.3 Create Dockerfile for embedding service
  - Use Python base image with sentence-transformers
  - Install gRPC dependencies
  - Configure model caching
  - Use multi-stage build for optimization
  - _Requirements: 21.1, 21.2, 21.3, 21.4_

- [ ]* 8.4 Write integration test for embedding service
  - Test Embed RPC returns correct dimension (384)
  - Test BatchEmbed processes multiple texts
  - Test FindSimilar returns semantically similar embeddings
  - _Requirements: 5.1, 5.2, 5.3, 5.5_

- [ ] 9. Implement Rust gRPC client for embedding service
- [ ] 9.1 Generate Rust code from Protocol Buffers
  - Use tonic-build in build.rs
  - Generate client stubs for EmbeddingService
  - _Requirements: 20.3, 20.4_

- [ ] 9.2 Create EmbeddingClient wrapper
  - Implement async embed() method
  - Implement batch_embed() for multiple texts
  - Implement find_similar() for similarity search
  - Add connection pooling and retry logic
  - Write doctests for EmbeddingClient usage
  - _Requirements: 20.3, 20.5, 39.1_

- [ ]* 9.3 Write integration test for Rust-Python communication
  - Test end-to-end embedding generation
  - Test type safety at service boundaries
  - Test error handling for service failures
  - _Requirements: 20.3, 20.4, 20.5_

## Phase 5: Requirements Parser

- [ ] 10. Implement requirements parser in cgc-parser crate
- [ ] 10.1 Create temporal keyword detector
  - Identify "when", "if", "then", "while" keywords
  - Extract premise and conclusion components
  - Handle nested conditionals
  - Write doctests for keyword detection
  - _Requirements: 1.2, 1.3, 2.3, 39.1_

- [ ] 10.2 Implement causal statement extractor
  - Parse natural language requirements text
  - Extract candidate causal statements
  - Assign confidence scores based on keyword presence
  - Write doctests for statement extraction
  - _Requirements: 1.1, 1.5, 39.1_

- [ ] 10.3 Implement constraint and assumption extractor
  - Identify constraint keywords (e.g., "must", "shall", "limit")
  - Identify assumption keywords (e.g., "assume", "given", "provided")
  - Extract context information
  - Write doctests for constraint/assumption extraction
  - _Requirements: 1.4, 39.1_

- [ ] 10.4 Implement logical decomposition
  - Decompose complex requirements into atomic P→Q relations
  - Preserve logical operators (AND, OR, NOT) in structure
  - Assign unique identifiers to decomposed statements
  - Maintain traceability to original requirements
  - Write doctests for decomposition
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 39.1_

- [ ]* 10.5 Write property test for temporal keyword recognition
  - **Property 9: Temporal Keyword Recognition**
  - **Validates: Requirements 1.1, 1.2, 1.3**

- [ ]* 10.6 Write property test for decomposition preserves logic
  - **Property 10: Decomposition Preserves Logic**
  - **Validates: Requirements 2.1, 2.2**

- [ ]* 10.7 Write property test for traceability round-trip
  - **Property 11: Traceability Round-Trip**
  - **Validates: Requirements 2.5, 11.3**


## Phase 6: Graph Construction Pipeline

- [ ] 11. Implement graph builder that integrates parser and embeddings
- [ ] 11.1 Create GraphBuilder orchestrator
  - Accept parsed causal statements
  - Request embeddings from embedding service for each proposition
  - Call CausalGraph.add_node for premises and conclusions
  - Call CausalGraph.add_edge to create relations
  - Write doctests for graph building
  - _Requirements: 4.1, 4.3, 4.6, 4.7, 39.1_

- [ ] 11.2 Implement constraint integration
  - Add constraint nodes to graph
  - Create ConstraintOn edges linking constraints to causal relations
  - Support multiple constraints per relation
  - Validate constraint linkage
  - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [ ] 11.3 Implement assumption integration
  - Add assumption nodes to graph
  - Create AssumptionFor edges linking assumptions to dependent relations
  - Track assumption dependencies
  - _Requirements: 1.4, 8.4_

- [ ] 11.4 Implement context integration
  - Add context nodes to graph
  - Create ContextualizedBy edges
  - Support contextual qualification of causal relations
  - _Requirements: 1.4_

- [ ]* 11.5 Write integration test for graph construction pipeline
  - Test end-to-end: requirements text → parsed statements → embeddings → graph
  - Verify nodes, edges, and metadata are correctly created
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4.1, 4.2, 4.3, 4.6, 4.7_

## Phase 7: Consistency Checker

- [ ] 12. Implement consistency checker in cgc-core
- [ ] 12.1 Create ConsistencyChecker for contradiction detection
  - Detect P→Q and P→¬Q patterns
  - Report contradicting edges with traceability
  - Write doctests for contradiction detection
  - _Requirements: 6.1, 6.3, 39.1_

- [ ] 12.2 Implement cycle detection
  - Use Tarjan's algorithm or DFS for cycle detection
  - Report all nodes in detected cycles
  - Write doctests for cycle detection
  - _Requirements: 6.2, 6.3, 39.1_

- [ ] 12.3 Implement constraint violation checker
  - Verify constraints are not violated by derivable conclusions
  - Report violated constraints with affected relations
  - _Requirements: 3.3, 6.5_

- [ ] 12.4 Implement graph validation marker
  - Mark graph as validated when consistency checks pass
  - Include validation timestamp
  - _Requirements: 6.4, 16.5_

- [ ]* 12.5 Write property test for contradiction detection
  - **Property 12: Contradiction Detection**
  - **Validates: Requirements 6.1**

- [ ]* 12.6 Write property test for cycle detection
  - **Property 13: Cycle Detection**
  - **Validates: Requirements 6.2**

## Phase 7.5: Semantic Refinement Engine

- [ ] 12.7 Implement semantic refinement engine in cgc-core
- [ ] 12.7.1 Create SemanticRefinementEngine structure
  - Define configurable thresholds: review_threshold (0.5), max_variance_threshold (0.8), aggregation_threshold (0.90)
  - Implement should_split() to check if node variance exceeds review threshold
  - Implement should_aggregate() to check if two nodes have similar centroids and edge patterns
  - Write doctests for threshold checking
  - _Requirements: 5.12, 5.13, 43.1, 43.5, 39.1_

- [ ] 12.7.2 Implement node splitting with clustering
  - Implement split_node() using k-means or hierarchical clustering
  - Partition embedding collection into coherent clusters
  - Create new nodes for each cluster with their own centroids
  - Redistribute edges based on semantic alignment
  - Maintain traceability of split operations
  - Write doctests for node splitting
  - _Requirements: 43.1, 43.2, 43.3, 43.4, 39.1_

- [ ] 12.7.3 Implement node aggregation
  - Implement aggregate_nodes() to merge two similar nodes
  - Merge embedding collections and recalculate centroid
  - Combine labels and remove duplicates
  - Merge edges and remove duplicates
  - Maintain traceability of aggregation operations
  - Write doctests for node aggregation
  - _Requirements: 43.5, 43.6, 43.7, 39.1_

- [ ] 12.7.4 Implement edge redistribution logic
  - Implement redistribute_edges() for split operations
  - Determine which cluster each edge aligns with semantically
  - Implement merge_edges() for aggregation operations
  - Remove duplicate edges after merging
  - _Requirements: 43.3, 43.6_

- [ ] 12.7.5 Implement refinement suggestion system
  - Implement suggest_refinements() to identify opportunities
  - Scan graph for high-variance nodes (split candidates)
  - Scan graph for similar centroids (aggregation candidates)
  - Generate RefinementSuggestion with rationale
  - Write doctests for suggestion generation
  - _Requirements: 43.10, 39.1_

- [ ] 12.7.6 Implement refinement execution with user approval
  - Create user approval interface for refinement operations
  - Execute split or aggregate operations after approval
  - Re-run consistency checking on affected subgraph
  - Notify user of changes with summary
  - _Requirements: 43.8, 43.9, 43.10_

- [ ]* 12.7.7 Write property test for variance threshold enforcement
  - **Property 23: Variance Threshold Enforcement**
  - **Validates: Requirements 5.13**

- [ ]* 12.7.8 Write property test for node split preserves edges
  - **Property 24: Node Split Preserves Edges**
  - **Validates: Requirements 43.3**

- [ ]* 12.7.9 Write property test for node aggregation preserves embeddings
  - **Property 25: Node Aggregation Preserves Embeddings**
  - **Validates: Requirements 43.6**

- [ ]* 12.7.10 Write property test for centroid drift convergence
  - **Property 26: Centroid Drift Convergence**
  - **Validates: Requirements 5.10**

- [ ]* 12.7.11 Write property test for split-aggregate idempotence
  - **Property 27: Split-Aggregate Idempotence**
  - **Validates: Requirements 43.1-43.7**

## Phase 8: Reasoning Engine

- [ ] 13. Implement reasoning engine in cgc-reasoning crate
- [ ] 13.1 Create ReasoningEngine with modus ponens
  - Implement apply_modus_ponens: given P and P→Q, derive Q
  - Support transitive inference through causal chains
  - Return all derivable conclusions from premises
  - Write doctests for modus ponens application
  - _Requirements: 8.2, 16.2, 39.1_

- [ ] 13.2 Implement test case generator
  - Generate test cases for P→Q relations
  - Include premise setup, action, and expected result
  - Add constraint checks to test cases
  - Add assumption validation to test preconditions
  - Output structured test format
  - Write doctests for test generation
  - _Requirements: 8.1, 8.3, 8.4, 8.5, 39.1_

- [ ] 13.3 Implement forward chaining for design exploration
  - Trace forward from design decisions through causal chains
  - Identify affected requirements
  - _Requirements: 10.2_

- [ ] 13.4 Implement backward chaining for "why" queries
  - Trace backward from conclusions to find all causal chains
  - Include assumptions and constraints in explanation
  - _Requirements: 12.1, 12.3_

- [ ]* 13.5 Write property test for modus ponens completeness
  - **Property 14: Modus Ponens Completeness**
  - **Validates: Requirements 8.2**

- [ ]* 13.6 Write property test for test case generation
  - **Property 15: Test Case Generation**
  - **Validates: Requirements 8.1**


## Phase 9: Serialization and Persistence

- [ ] 14. Implement graph serialization
- [ ] 14.1 Implement JSON serialization with serde
  - Serialize complete graph structure with nodes, edges, embeddings
  - Include all metadata and traceability information
  - Produce valid JSON output
  - Write doctests for serialization
  - _Requirements: 17.1, 17.2, 4.8, 39.1_

- [ ] 14.2 Implement JSON deserialization
  - Reconstruct graph from JSON
  - Rebuild embedding index for similarity queries
  - Restore all relationships and metadata
  - Write doctests for deserialization
  - _Requirements: 17.3, 17.4, 39.1_

- [ ] 14.3 Add support for GraphML format
  - Implement GraphML serialization with embedding preservation
  - Support standard GraphML structure
  - _Requirements: 17.5_

- [ ] 14.4 Add support for RDF format
  - Implement RDF serialization with embedding preservation
  - Use appropriate ontology for causal relations
  - _Requirements: 17.5_

## Phase 10: Graph Update Operations

- [ ] 15. Implement real-time graph updates
- [ ] 15.1 Implement update_requirement operation
  - Identify affected nodes and edges
  - Update without full graph rebuild
  - Re-run consistency checking on affected subgraph
  - Write doctests for updates
  - _Requirements: 9.1, 9.5, 39.1_

- [ ] 15.2 Implement add_requirement operation
  - Integrate new causal relations into existing graph
  - Merge with similar existing nodes
  - _Requirements: 9.2, 9.3_

- [ ] 15.3 Implement remove_requirement operation
  - Remove corresponding edges
  - Preserve nodes referenced by other requirements
  - _Requirements: 9.4_

- [ ] 15.4 Implement update notification system
  - Notify user of new inconsistencies after updates
  - Report validation issues
  - _Requirements: 9.6_

- [ ]* 15.5 Write integration test for real-time updates
  - Test add, update, remove operations
  - Verify consistency checking on updates
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

## Phase 11: Traceability System

- [ ] 16. Implement comprehensive traceability
- [ ] 16.1 Create TraceabilityMap structure
  - Map nodes to source requirements
  - Map requirements to derived graph elements
  - Support bidirectional queries
  - Write doctests for traceability queries
  - _Requirements: 11.1, 11.2, 11.3, 39.1_

- [ ] 16.2 Implement impact analysis
  - Identify downstream graph elements affected by requirement changes
  - Trace complete chain from requirement to graph structure
  - _Requirements: 11.4, 11.5_

- [ ] 16.3 Implement traceability report generation
  - Generate reports showing requirement-to-graph mappings
  - Include complete transformation chain
  - _Requirements: 11.5_

- [ ]* 16.4 Write integration test for traceability
  - Test bidirectional traceability queries
  - Test impact analysis
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

## Phase 12: Query System

- [ ] 17. Implement natural language query interface
- [ ] 17.1 Create QueryEngine for "why" queries
  - Parse "why does Q occur" queries
  - Return all causal chains leading to Q
  - Include premises, assumptions, and constraints
  - Write doctests for "why" queries
  - _Requirements: 12.1, 12.3, 39.1_

- [ ] 17.2 Implement "what" queries for consequences
  - Parse "what are consequences of P" queries
  - Apply modus ponens to derive all conclusions
  - Return derivable conclusions with causal chains
  - Write doctests for "what" queries
  - _Requirements: 12.2, 12.3, 39.1_

- [ ] 17.3 Implement design rationale queries
  - Trace back to original requirements
  - Explain justification for causal relations
  - _Requirements: 12.4_

- [ ] 17.4 Implement natural language response generation
  - Convert graph results to natural language
  - Include references to formal graph structure
  - _Requirements: 12.5_

- [ ]* 17.5 Write integration test for query system
  - Test "why" and "what" queries
  - Test natural language response generation
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_


## Phase 13: Design-Space Exploration

- [ ] 18. Implement design-space exploration features
- [ ] 18.1 Create DesignExplorer for alternative evaluation
  - Evaluate which requirements are satisfied by design alternative
  - Trace forward through causal chains
  - Write doctests for design exploration
  - _Requirements: 10.1, 10.2, 39.1_

- [ ] 18.2 Implement alternative comparison
  - Compare multiple design alternatives
  - Show requirements coverage for each
  - _Requirements: 10.3_

- [ ] 18.3 Implement conflict detection
  - Identify requirements that cannot be simultaneously satisfied
  - Report conflicting requirements
  - _Requirements: 10.4_

- [ ] 18.4 Implement alternative ranking
  - Rank alternatives by requirements coverage
  - Consider constraint satisfaction
  - _Requirements: 10.5_

- [ ]* 18.5 Write integration test for design-space exploration
  - Test alternative evaluation and comparison
  - Test conflict detection and ranking
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

## Phase 14: Extensibility Framework

- [ ] 19. Implement extensibility mechanisms
- [ ] 19.1 Create custom constraint type registry
  - Allow registration of custom constraint types
  - Apply custom constraints during integration
  - Write doctests for custom constraints
  - _Requirements: 13.1, 39.1_

- [ ] 19.2 Create custom inference rule registry
  - Allow registration of custom inference rules
  - Incorporate rules into reasoning engine
  - Write doctests for custom rules
  - _Requirements: 13.2, 39.1_

- [ ] 19.3 Create custom parser pattern registry
  - Allow registration of domain-specific pattern matchers
  - Extend parser with custom patterns
  - Write doctests for custom patterns
  - _Requirements: 13.3, 39.1_

- [ ] 19.4 Implement extension validation
  - Validate custom extensions don't break existing functionality
  - Run regression tests on extension addition
  - _Requirements: 13.4_

- [ ] 19.5 Implement backward compatibility checks
  - Ensure existing graphs remain compatible
  - Version extension APIs
  - _Requirements: 13.5_

- [ ]* 19.6 Write integration test for extensibility
  - Test custom constraint types
  - Test custom inference rules
  - Test custom parser patterns
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

## Phase 15: Robustness and Error Handling

- [ ] 20. Implement robust error handling
- [ ] 20.1 Create ambiguity detection in parser
  - Flag ambiguous language
  - Request clarification from user
  - _Requirements: 15.1_

- [ ] 20.2 Implement incomplete requirements detection
  - Identify missing information
  - Suggest what is needed
  - _Requirements: 15.2_

- [ ] 20.3 Implement graceful parsing failure handling
  - Continue processing other requirements on failure
  - Report failures with context
  - _Requirements: 15.3_

- [ ] 20.4 Implement low-confidence marking
  - Mark low-confidence causal relations for review
  - Provide confidence thresholds
  - _Requirements: 15.4_

- [ ] 20.5 Implement contradiction resolution interface
  - Present contradictions to user
  - Provide resolution options
  - _Requirements: 15.5_

- [ ]* 20.6 Write integration test for robustness
  - Test ambiguous input handling
  - Test incomplete requirements handling
  - Test parsing failure recovery
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

## Phase 16: Formal Logic Validation

- [ ] 21. Implement formal logic validation
- [ ] 21.1 Create LogicValidator for inference pattern checking
  - Verify modus ponens, modus tollens, and other patterns
  - Check all causal relations follow valid inference
  - Write doctests for logic validation
  - _Requirements: 16.1, 16.2, 39.1_

- [ ] 21.2 Implement invalid inference detection
  - Detect and report logical errors
  - Provide affected graph elements
  - _Requirements: 16.3_

- [ ] 21.3 Implement constraint violation validation
  - Verify constraints not violated by derivable conclusions
  - _Requirements: 16.4_

- [ ] 21.4 Implement graph certification
  - Certify logically sound graphs
  - Add validation timestamp
  - _Requirements: 16.5_

- [ ]* 21.5 Write integration test for logic validation
  - Test inference pattern validation
  - Test invalid inference detection
  - Test constraint violation checking
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_


## Phase 17: CLI Implementation

- [ ] 22. Implement comprehensive CLI in cgc-cli crate
- [ ] 22.1 Create CLI structure with clap
  - Define commands: import, build, query, validate, test-gen, export
  - Support stdin/stdout for pipeline integration
  - Implement proper exit codes
  - Write doctests for CLI usage
  - _Requirements: 29.1, 29.2, 29.4, 29.5, 30.2, 39.1_

- [ ] 22.2 Implement import command
  - Import requirements from CSV files
  - Support stdin input with "-" argument
  - Output structured JSON
  - _Requirements: 27.1, 27.2, 27.3, 27.4, 29.2_

- [ ] 22.3 Implement build command
  - Build causal graph from requirements
  - Output graph in JSON format
  - Support pipeline input
  - _Requirements: 29.2, 30.4_

- [ ] 22.4 Implement query command
  - Support "why" and "what" queries
  - Output results in JSON/plain text
  - _Requirements: 29.2, 29.3_

- [ ] 22.5 Implement validate command
  - Run consistency checks on graph
  - Output validation results
  - _Requirements: 29.2_

- [ ] 22.6 Implement test-gen command
  - Generate test cases from graph edges
  - Output structured test format
  - _Requirements: 29.2_

- [ ] 22.7 Implement export command
  - Export graph in multiple formats (JSON, GraphML, RDF)
  - Support stdout output
  - _Requirements: 14.4, 29.2_

- [ ] 22.8 Implement refine command
  - Suggest semantic refinement operations (split/aggregate)
  - Display suggestions with rationale
  - Support --auto flag for automatic approval
  - Support --interactive flag for user approval
  - Output refinement summary
  - _Requirements: 43.9, 43.10, 29.2_

- [ ] 22.9 Implement Unix composability features
  - Format output for grep, awk, sed, jq compatibility
  - Support streaming for large datasets
  - Write errors to stderr, data to stdout
  - _Requirements: 30.1, 30.3, 30.4, 30.5_

- [ ]* 22.10 Write integration test for CLI commands
  - Test all commands with various inputs
  - Test pipeline composition
  - Test error handling and exit codes
  - Test refine command with suggestions and approval
  - _Requirements: 29.1, 29.2, 29.3, 29.4, 29.5, 30.1, 30.2, 30.3, 30.4, 30.5, 43.10_

## Phase 18: Web UI Implementation

- [ ] 23. Implement web UI in cgc-web crate
- [ ] 23.1 Create web server with axum
  - Set up HTTP server on localhost
  - Serve static files for UI
  - Implement API endpoints for UI operations
  - _Requirements: 26.1, 26.2_

- [ ] 23.2 Implement CSV import UI
  - Create file browser for local CSV selection
  - Display CSV preview before import
  - Show import progress
  - _Requirements: 26.3, 26.4, 27.1, 27.2_

- [ ] 23.3 Implement graph visualization
  - Create interactive graph visualization with zoom/pan
  - Support node selection and highlighting
  - Provide layout options (hierarchical, layered, clustered)
  - _Requirements: 26.5, 14.1_

- [ ] 23.4 Implement chat interface for queries
  - Create text input for natural language queries
  - Display query results with natural language explanations
  - Add clickable node references that highlight in visualization
  - _Requirements: 26.6, 26.7, 26.8_

- [ ] 23.5 Implement UI responsiveness features
  - Add loading indicators for operations > 1 second
  - Implement progress bars for long operations
  - Add hover states and visual feedback
  - Support responsive layout for different window sizes
  - _Requirements: 33.2, 33.4, 33.5_

- [ ] 23.6 Implement error display
  - Show clear, actionable error messages
  - Provide error context and recovery suggestions
  - _Requirements: 33.3_

- [ ] 23.7 Implement semantic refinement UI
  - Display refinement suggestions (split/aggregate opportunities)
  - Show rationale for each suggestion (variance, similarity scores)
  - Provide approve/reject buttons for each suggestion
  - Display before/after preview of refinement operations
  - Show notification of completed refinements with summary
  - _Requirements: 43.9, 43.10_

- [ ]* 23.8 Write integration test for web UI
  - Test CSV import flow
  - Test graph visualization rendering
  - Test chat query interface
  - Test semantic refinement suggestion and approval flow
  - _Requirements: 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7, 26.8, 43.10_

## Phase 19: Local Storage Management

- [ ] 24. Implement local storage system
- [ ] 24.1 Create storage directory structure
  - Create requirements/, graphs/, config/ folders
  - Initialize on system installation
  - Write doctests for storage initialization
  - _Requirements: 28.1, 24.5, 39.1_

- [ ] 24.2 Implement requirements storage
  - Store imported requirements with metadata
  - Include import timestamp and source file
  - _Requirements: 27.4, 28.2_

- [ ] 24.3 Implement graph storage
  - Store serialized graphs in graphs/ folder
  - Organize by project or timestamp
  - _Requirements: 28.3_

- [ ] 24.4 Implement storage browser for UI
  - Display folder structure in UI
  - Allow browsing and selecting requirements/graphs
  - _Requirements: 28.4, 28.5_

- [ ]* 24.5 Write integration test for storage system
  - Test directory creation
  - Test requirements and graph storage
  - Test storage browsing
  - _Requirements: 28.1, 28.2, 28.3, 28.4, 28.5_


## Phase 20: Docker Containerization

- [ ] 25. Create Docker containers for all services
- [ ] 25.1 Create Dockerfile for cgc-core service
  - Use Rust base image with multi-stage build
  - Include all Rust crates (core, parser, reasoning, cli, web)
  - Minimize image size with build optimization
  - _Requirements: 21.1, 21.2, 21.3, 21.4_

- [ ] 25.2 Create docker-compose.yml
  - Define cgc-core, cgc-web, and embedding-service
  - Configure service dependencies and networks
  - Set up volume mounts for data persistence
  - Use environment variables for configuration
  - Enable service discovery by name
  - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5_

- [ ] 25.3 Implement local Docker deployment
  - Ensure all services run on local Docker daemon
  - Bind to localhost ports
  - Persist data to local Docker volumes
  - _Requirements: 25.1, 25.2, 25.3, 25.4, 25.5_

- [ ]* 25.4 Write integration test for Docker deployment
  - Test docker-compose up starts all services
  - Test service communication
  - Test data persistence across restarts
  - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 25.1, 25.2, 25.3, 25.4, 25.5_

## Phase 21: System Installer

- [ ] 26. Create OS-specific installers
- [ ] 26.1 Create Linux installer script
  - Check for Docker and Docker Compose
  - Install Docker if missing (with user permission)
  - Set up application and data directories
  - Add CLI to PATH
  - _Requirements: 24.1, 24.4, 24.5, 24.6_

- [ ] 26.2 Create Windows installer script
  - Check for Docker Desktop
  - Install Docker Desktop if missing (with user permission)
  - Set up application and data directories
  - Add CLI to PATH
  - _Requirements: 24.2, 24.4, 24.5, 24.6_

- [ ] 26.3 Create macOS installer script
  - Check for Docker Desktop
  - Install Docker Desktop if missing (with user permission)
  - Set up application and data directories
  - Add CLI to PATH
  - _Requirements: 24.3, 24.4, 24.5, 24.6_

- [ ]* 26.4 Write integration test for installer
  - Test installer on clean system (in VM/container)
  - Verify Docker setup
  - Verify directory creation
  - Verify CLI availability
  - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5, 24.6_

## Phase 22: Visualization and Interpretation

- [ ] 27. Implement graph visualization and interpretation features
- [ ] 27.1 Implement visualization layout algorithms
  - Hierarchical layout for causal chains
  - Layered layout for complex graphs
  - Clustered layout for related nodes
  - _Requirements: 14.1_

- [ ] 27.2 Implement human-readable labeling
  - Display natural language labels from requirements
  - Show relation types on edges
  - _Requirements: 14.2_

- [ ] 27.3 Implement causal relation explanation generator
  - Generate natural language descriptions of P→Q relations
  - Include context, assumptions, and constraints
  - Write doctests for explanation generation
  - _Requirements: 14.3, 39.1_

- [ ] 27.4 Implement export formats
  - Export visual diagrams (SVG, PNG)
  - Export textual reports (Markdown, HTML)
  - Export structured data (JSON, GraphML, RDF)
  - _Requirements: 14.4_

- [ ] 27.5 Implement critical path highlighting
  - Identify and highlight critical causal chains
  - Show high-impact relations
  - _Requirements: 14.5_

- [ ]* 27.6 Write integration test for visualization
  - Test layout algorithms
  - Test export formats
  - Test critical path identification
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

## Phase 23: Performance Optimization

- [ ] 28. Implement performance optimizations and benchmarks
- [ ] 28.1 Optimize CSV import performance
  - Implement parallel parsing for large files
  - Add streaming for memory efficiency
  - Target: 1000 requirements in < 5 seconds
  - _Requirements: 31.1_

- [ ] 28.2 Optimize graph construction performance
  - Batch embedding requests to reduce latency
  - Parallelize node creation where possible
  - Optimize centroid calculations for large embedding collections
  - Target: 1000 requirements → graph in < 10 seconds
  - _Requirements: 31.2_

- [ ] 28.3 Optimize query performance
  - Use CSR graph for efficient traversal
  - Cache frequently accessed paths
  - Target: queries in < 500ms
  - _Requirements: 31.3_

- [ ] 28.4 Optimize UI load time
  - Lazy load graph visualization
  - Minimize initial bundle size
  - Target: UI loads in < 2 seconds
  - _Requirements: 31.4_

- [ ] 28.5 Optimize visualization rendering
  - Use WebGL for large graphs
  - Implement level-of-detail rendering
  - Target: 500 nodes rendered in < 3 seconds
  - _Requirements: 31.5_

- [ ]* 28.6 Write property test for import performance
  - **Property 18: Import Performance**
  - **Validates: Requirements 31.1**

- [ ]* 28.7 Write property test for graph construction performance
  - **Property 19: Graph Construction Performance**
  - **Validates: Requirements 31.2**

- [ ]* 28.8 Write property test for query response time
  - **Property 20: Query Response Time**
  - **Validates: Requirements 31.3**


## Phase 24: Scalability Testing

- [ ] 29. Implement scalability features and tests
- [ ] 29.1 Test and optimize for 2,500 requirements
  - Verify graph construction in < 25 seconds
  - _Requirements: 32.1_

- [ ] 29.2 Test and optimize for 5,000 requirements
  - Verify graph construction in < 50 seconds
  - _Requirements: 32.2_

- [ ] 29.3 Test and optimize for 10,000 requirements
  - Verify graph construction in < 120 seconds
  - Ensure no memory exhaustion
  - _Requirements: 32.3_

- [ ] 29.4 Optimize for 10,000+ node graphs
  - Verify query and traversal performance
  - Use HNSW for sub-linear similarity search
  - _Requirements: 32.4, 32.5_

- [ ] 29.5 Optimize memory usage
  - Target: < 500MB for idle background services
  - Use memory profiling to identify leaks
  - _Requirements: 32.6_

- [ ] 29.6 Implement progress indicators
  - Show progress for operations > 2 seconds
  - Display estimated time remaining
  - _Requirements: 32.7_

- [ ]* 29.7 Write property test for scalability
  - **Property 21: Scalability**
  - **Validates: Requirements 32.1, 32.2, 32.3**

- [ ]* 29.8 Write property test for memory efficiency
  - **Property 22: Memory Efficiency**
  - **Validates: Requirements 32.6**

## Phase 25: Documentation and Code Traceability

- [ ] 30. Implement comprehensive documentation
- [ ] 30.1 Add requirement references to all functions
  - Document which requirement each function implements
  - Use format: "Implements: Requirement X, AC X.Y"
  - Apply to all public and private functions
  - _Requirements: 34.1, 34.2, 34.3, 34.4_

- [ ] 30.2 Add module-level documentation
  - Explain module purpose and relationship to requirements
  - Include usage examples
  - _Requirements: 41.1_

- [ ] 30.3 Add struct and enum documentation
  - Document purpose, fields, and usage
  - Include examples
  - _Requirements: 41.2_

- [ ] 30.4 Ensure all public functions have doctests
  - Verify at least one doctest per public function
  - Cover multiple use cases where applicable
  - Ensure doctests are executable
  - _Requirements: 39.1, 39.2, 39.3, 39.4, 39.5_

- [ ] 30.5 Generate HTML documentation
  - Use cargo doc to generate documentation
  - Ensure all examples are tested
  - _Requirements: 41.4, 41.5_

- [ ] 30.6 Implement searchable requirement references
  - Enable searching codebase by requirement number
  - Document search patterns for developers
  - _Requirements: 34.5_

## Phase 26: Refactoring Support

- [ ] 31. Implement refactoring support mechanisms
- [ ] 31.1 Use Rust deprecation attributes
  - Mark deprecated functions with #[deprecated]
  - Provide migration guidance in deprecation messages
  - _Requirements: 40.3_

- [ ] 31.2 Document function usage tracking
  - Use rust-analyzer for finding references
  - Document refactoring procedures
  - _Requirements: 40.1, 40.4_

- [ ] 31.3 Implement drift detection
  - Use failing doctests to identify drift
  - Use failing unit tests to identify specification drift
  - _Requirements: 40.5_

## Phase 27: Final Integration and System Testing

- [ ] 32. Final integration and system testing
- [ ] 32.1 Run complete end-to-end system test
  - Import requirements → build graph → query → export
  - Test all CLI commands in pipeline
  - Test UI workflow
  - _Requirements: All_

- [ ] 32.2 Verify all performance targets
  - Run performance benchmarks
  - Verify all targets are met
  - _Requirements: 31.1, 31.2, 31.3, 31.4, 31.5, 32.1, 32.2, 32.3, 32.4, 32.5, 32.6_

- [ ] 32.3 Verify all property tests pass
  - Run all 19 property tests
  - Ensure 100+ iterations each
  - _Requirements: All correctness properties_

- [ ] 32.4 Verify test coverage meets thresholds
  - Overall coverage ≥80%
  - Critical paths ≥90%
  - Generate coverage reports
  - _Requirements: 42.1, 42.2, 42.3, 42.4, 42.5_

- [ ] 32.5 Generate final documentation package
  - Build complete cargo doc
  - Generate coverage reports
  - Create user guide and API documentation
  - _Requirements: 41.1, 41.2, 41.3, 41.4, 41.5_

