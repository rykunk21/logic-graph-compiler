//! Temporal keyword detection for causal statement extraction.
//!
//! Implements: Requirements 1.2, 1.3

/// Temporal keywords that indicate causal relationships
pub const TEMPORAL_KEYWORDS: &[&str] = &[
    "when", "if", "then", "while", "whenever", "after", "before", "during",
];

/// Check if text contains temporal keywords
pub fn contains_temporal_keywords(text: &str) -> bool {
    let text_lower = text.to_lowercase();
    TEMPORAL_KEYWORDS
        .iter()
        .any(|&keyword| text_lower.contains(keyword))
}
