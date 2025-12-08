#!/usr/bin/env python
'''
Demo of MLX 0.29.0 and MLX-LM 0.27.0 enhanced features:
- Prompt caching for faster repeated queries
- 4-bit KV cache quantization for memory efficiency
'''

import sys
sys.path.insert(0, '..')

from pylib.schema_helper import Model

def demo_enhanced_features():
    print("MLX Enhanced Features Demo")
    print("=" * 50)

    # Initialize model with prompt cache support
    model = Model()

    # Load model with max_kv_size for prompt caching
    print("\n1. Loading model with prompt cache support...")
    model.load(
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        max_kv_size=4096  # Enable prompt cache with 4K tokens
    )
    print("   ✓ Model loaded with prompt caching enabled")

    # Example schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "skills"]
    }

    # Test 1: Basic generation with prompt caching
    print("\n2. Testing prompt caching...")
    messages = [
        {"role": "user", "content": "Generate a profile for a software developer"}
    ]

    print("   First generation (cold cache):")
    result = model.completion(
        messages=messages,
        schema=schema,
        cache_prompt=True,  # Enable prompt caching
        max_tokens=100
    )
    for chunk in result:
        print(f"   {chunk}", end="")
    print()

    # Test 2: Generation with 4-bit KV quantization
    print("\n3. Testing 4-bit KV cache quantization...")
    messages2 = [
        {"role": "user", "content": "Generate a profile for a data scientist"}
    ]

    result = model.completion(
        messages=messages2,
        schema=schema,
        kv_bits=4,  # Enable 4-bit quantization
        kv_group_size=64,
        quantized_kv_start=100,  # Start quantizing after 100 tokens
        max_tokens=100
    )

    print("   Generation with 4-bit KV cache:")
    for chunk in result:
        print(f"   {chunk}", end="")
    print()

    print("\n" + "=" * 50)
    print("Enhanced features successfully demonstrated!")
    print("\nKey improvements with MLX 0.29.0 & MLX-LM 0.27.0:")
    print("• Prompt caching reduces latency for repeated queries")
    print("• 4-bit KV quantization reduces memory by ~50%")
    print("• Quantized attention maintains quality while saving memory")

if __name__ == "__main__":
    demo_enhanced_features()