# Causal Graph Compiler

A requirements compiler that transforms natural-language engineering requirements into structured causal logic graphs for automated Model-Based Systems Engineering (MBSE), design-space exploration, traceability, and test generation.

## Overview

The Causal Graph Compiler translates natural-language requirements into formal causal logic graphs representing P→Q (premise-to-conclusion) relations, assumptions, constraints, and context. The system enables:

- **Automated reasoning** via logical inference (modus ponens, transitive chains)
- **Design-space exploration** by evaluating alternatives against requirements
- **Comprehensive traceability** from requirements to graph elements
- **Automated test generation** from causal relations
- **Consistency checking** for contradictions and circular dependencies
- **Natural language queries** to understand system behavior and design rationale

### Key Features

- **Embedding-based node identity**: Semantic embeddings prevent duplicate nodes for equivalent statements
- **Role fluidity**: Nodes can serve as both premises and conclusions in different causal chains
- **CSR graph structure**: Memory-efficient compressed sparse row format for scalability
- **Local-first**: All processing occurs locally without cloud dependencies
- **Polyglot architecture**: Rust backbone with Python services for ML/NLP tasks
- **CLI-first**: Unix-composable command-line interface for automation
- **Simple web UI**: Minimal interface for CSV import and graph visualization

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interfaces                               │
├──────────────────────────┬──────────────────────────────────────┤
│   CLI (Rust)             │   Web UI (Rust + WASM)              │
│   - Unix composable      │   - CSV import                       │
│   - JSON/CSV output      │   - Graph visualization              │
│   - stdin/stdout         │   - Natural language queries         │
└──────────────────────────┴──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Core Graph Engine (Rust)                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  CSR Graph                                                 │ │
│  │  - Nodes: embedding vectors + labels                       │ │
│  │  - Edges: relation types + metadata                        │ │
│  │  - O(V+E) space, O(degree) edge iteration                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Embedding Index (HNSW)                                    │ │
│  │  - Maps embeddings → NodeIds                               │ │
│  │  - O(log n) similarity search                              │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (gRPC + Protocol Buffers)
┌─────────────────────────────────────────────────────────────────┐
│              Specialized Services (Python)                       │
│  ┌──────────────────────────────┐  ┌────────────────────────┐  │
│  │ Embedding Service            │  │ NLP Service            │  │
│  │ - sentence-transformers      │  │ - spaCy                │  │
│  │ - all-MiniLM-L6-v2          │  │ - Pattern matching     │  │
│  └──────────────────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- **Docker** and **Docker Compose** (installer will set up if missing)
- **Linux**, **Windows**, or **macOS**

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/causal-graph-compiler/main/installer/linux/install.sh | bash
```

### Windows

```powershell
iwr -useb https://raw.githubusercontent.com/yourusername/causal-graph-compiler/main/installer/windows/install.ps1 | iex
```

### macOS

```bash
curl -fsSL https://raw.githubusercontent.com/yourusername/causal-graph-compiler/main/installer/macos/install.sh | bash
```

The installer will:
1. Check for Docker and install if needed (with your permission)
2. Set up the application and data directories
3. Add the `cgc` CLI to your PATH
4. Start the containerized services

## Quick Start

### 1. Import Requirements from CSV

```bash
cgc import requirements.csv -o requirements.json
```

CSV format:
```csv
id,requirement_text
1,"WHEN temperature exceeds 100°C, THEN the cooling system SHALL activate"
2,"WHEN cooling system activates, THEN temperature SHALL decrease"
```

### 2. Build Causal Graph

```bash
cgc build requirements.json -o graph.json
```

### 3. Query the Graph

```bash
# Why does something occur?
cgc query graph.json "why does temperature decrease?"

# What are the consequences?
cgc query graph.json "what happens when temperature exceeds 100°C?"
```

### 4. Validate Consistency

```bash
cgc validate graph.json
```

### 5. Generate Test Cases

```bash
cgc test-gen graph.json --edge-id <edge-id> -o tests.json
```

### 6. Export Graph

```bash
# Export as JSON
cgc export graph.json -f json -o output.json

# Export as GraphML
cgc export graph.json -f graphml -o output.graphml

# Export as RDF
cgc export graph.json -f rdf -o output.rdf
```

## CLI Usage

### Unix Composability

The CLI is designed for Unix pipelines:

```bash
# Pipeline example
cat requirements.csv | cgc import - | cgc build - | cgc validate -

# Filter and process
cgc query graph.json "why" | jq '.causal_chains[] | select(.confidence > 0.8)'

# Batch processing
find . -name "*.csv" | xargs -I {} cgc import {} -o {}.json
```

### Commands

- `cgc import <file>` - Import requirements from CSV
- `cgc build <file>` - Build causal graph from requirements
- `cgc query <graph> <query>` - Query the graph
- `cgc validate <graph>` - Check consistency
- `cgc test-gen <graph>` - Generate test cases
- `cgc export <graph>` - Export in various formats

All commands support:
- Reading from stdin with `-` as filename
- JSON output for structured data
- Proper exit codes (0=success, 1=error)
- Errors to stderr, data to stdout

## Web UI

Start the web interface:

```bash
docker-compose up
```

Then open http://localhost:3000 in your browser.

Features:
- **CSV Import**: Browse and import local CSV files
- **Graph Visualization**: Interactive graph with zoom, pan, and node selection
- **Natural Language Queries**: Chat interface for asking questions about the graph
- **Export**: Download graphs in various formats

## Development

### Project Structure

```
causal-graph-compiler/
├── Cargo.toml                 # Workspace root
├── docker-compose.yml
├── proto/                     # Protocol Buffer definitions
│   ├── embedding_service.proto
│   └── nlp_service.proto
│
├── crates/                    # Rust packages
│   ├── cgc-core/             # Core graph engine
│   ├── cgc-parser/           # Requirements parser
│   ├── cgc-reasoning/        # Reasoning engine
│   ├── cgc-cli/              # CLI application
│   └── cgc-web/              # Web server
│
├── services/                  # Python services
│   ├── embedding-service/
│   └── nlp-service/
│
├── installer/                 # OS-specific installers
│   ├── linux/install.sh
│   ├── windows/install.ps1
│   └── macos/install.sh
│
└── tests/                     # Integration tests
    ├── integration/
    └── property/
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-graph-compiler.git
cd causal-graph-compiler

# Build all Rust crates
cargo build --release

# Build Docker containers
docker-compose build

# Run tests
cargo test
```

### Test-Driven Development

This project follows strict TDD:

1. **Write tests first** (red phase)
2. **Implement to pass tests** (green phase)
3. **Refactor while maintaining tests** (refactor phase)

Test hierarchy:
- **Integration tests**: One per requirement (verify complete user stories)
- **Unit tests**: One per acceptance criterion (verify specific functionality)
- **Property tests**: Verify universal properties across all inputs (100+ iterations)
- **Doctests**: Executable examples in documentation

### Running Tests

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test '*'

# Run property tests
cargo test --test property

# Run with coverage
cargo tarpaulin --out Html
```

### Documentation

Generate and view documentation:

```bash
cargo doc --open
```

All public functions include:
- Purpose and usage description
- Requirement references (e.g., "Implements: Requirement 5, AC 5.3")
- Executable doctests with examples

## Performance

Target performance metrics:

- **Import**: 1000 requirements in < 5 seconds
- **Graph construction**: 1000 requirements → graph in < 10 seconds
- **Queries**: Results in < 500ms
- **UI load**: < 2 seconds
- **Visualization**: 500 nodes rendered in < 3 seconds

Scalability:
- 2,500 requirements: < 25 seconds
- 5,000 requirements: < 50 seconds
- 10,000 requirements: < 120 seconds
- Memory: < 500MB idle

## Technology Stack

### Rust Components
- `csr` - Compressed Sparse Row graph
- `serde` / `serde_json` - Serialization
- `tonic` - gRPC framework
- `prost` - Protocol Buffers
- `tokio` - Async runtime
- `axum` - Web framework
- `clap` - CLI parsing
- `proptest` - Property-based testing

### Python Components
- `sentence-transformers` - Semantic embeddings
- `hnswlib` - Approximate nearest neighbor
- `grpcio` - gRPC framework
- `spacy` - NLP (optional)

## Contributing

1. Read the [requirements document](.kiro/specs/causal-graph-compiler/requirements.md)
2. Review the [design document](.kiro/specs/causal-graph-compiler/design.md)
3. Check the [task list](.kiro/specs/causal-graph-compiler/tasks.md)
4. Follow TDD: write tests before implementation
5. Ensure all tests pass and coverage meets thresholds (≥80% overall, ≥90% critical paths)
6. Add requirement references to all functions
7. Include doctests for all public functions

## License

[Your License Here]

## References

- [Requirements Document](.kiro/specs/causal-graph-compiler/requirements.md) - Complete system requirements
- [Design Document](.kiro/specs/causal-graph-compiler/design.md) - Architecture and design decisions
- [Task List](.kiro/specs/causal-graph-compiler/tasks.md) - Implementation plan
