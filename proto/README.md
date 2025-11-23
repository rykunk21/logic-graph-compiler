# Protocol Buffer Definitions

This directory contains Protocol Buffer (.proto) files defining the interface contracts between Rust and Python services.

## Files

- `embedding_service.proto` - Embedding generation service interface
  - `Embed` - Generate embedding for single text
  - `BatchEmbed` - Generate embeddings for multiple texts
  - `FindSimilar` - Find similar embeddings using HNSW
  
- `nlp_service.proto` - NLP service interface (optional)
  - `ExtractEntities` - Named entity recognition
  - `ParseDependencies` - Dependency parsing
  - `Tokenize` - Text tokenization

## Generating Code

### Rust

Code generation is handled automatically by `tonic-build` in the build.rs files.

Create `build.rs` in the crate that needs gRPC:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../../proto/embedding_service.proto")?;
    Ok(())
}
```

Then use in your code:

```rust
pub mod embedding {
    tonic::include_proto!("embedding");
}
```

### Python

Install dependencies:
```bash
pip install grpcio grpcio-tools
```

Generate Python code:
```bash
cd proto
python -m grpc_tools.protoc -I. --python_out=../services/embedding-service --grpc_python_out=../services/embedding-service embedding_service.proto
python -m grpc_tools.protoc -I. --python_out=../services/nlp-service --grpc_python_out=../services/nlp-service nlp_service.proto
```

## Type Safety

Protocol Buffers provide type safety at the service boundary:

- **Compile-time checking** in Rust via `tonic`
- **Runtime validation** in Python via `grpcio`
- **Schema evolution** support with field numbers
- **Cross-language compatibility** guaranteed by protobuf

## Schema Validation

The proto files enforce:
- Required vs optional fields
- Field types (string, int32, float, repeated)
- Message structure
- Service method signatures

Any type mismatch will be caught:
- At compile time in Rust
- At runtime in Python with clear error messages

Implements: Requirements 20.3, 20.4, 20.5
