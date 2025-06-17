#!/usr/bin/env python3
"""
Check current embedding provider status and compatibility.

This script displays information about the current embedding provider
configuration and any stored embedding metadata.

Usage:
    python tests/check_embeddings.py
"""
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.embeddings_provider import EmbeddingsProvider
from config import config


def main():
    """Check embedding provider status."""
    marker_path = Path(EmbeddingsProvider.MARKER_FILE)
    
    print("="*60)
    print("EMBEDDING PROVIDER STATUS")
    print("="*60)
    
    # Show current configuration
    print(f"Current config:")
    print(f"  Provider: {config.embeddings.provider}")
    if config.embeddings.provider == "local":
        print(f"  Model: BAAI/bge-large-en-v1.5")
    else:
        print(f"  Model: {config.embeddings.remote.model}")
    
    # Check stored marker
    if marker_path.exists():
        try:
            stored = json.loads(marker_path.read_text())
            print(f"\nStored embeddings:")
            print(f"  Provider: {stored['provider']}")
            print(f"  Model: {stored['model']}")
            print(f"  Dimension: {stored['dimension']}")
            print(f"  Created: {stored['created_at']}")
            print(f"  Version: {stored['version']}")
            
            # Check compatibility
            current_id = f"{config.embeddings.provider}:BAAI/bge-large-en-v1.5" if config.embeddings.provider == "local" else f"{config.embeddings.provider}:{config.embeddings.remote.model}"
            stored_id = f"{stored['provider']}:{stored['model']}"
            
            if current_id == stored_id:
                print(f"\n✅ Status: COMPATIBLE")
                print(f"   Current config matches stored embeddings")
            else:
                print(f"\n❌ Status: INCOMPATIBLE")
                print(f"   Current: {current_id}")
                print(f"   Stored:  {stored_id}")
                print(f"   Action needed: Run migration script")
                
        except json.JSONDecodeError:
            print(f"\n❌ Status: CORRUPTED")
            print(f"   Marker file is corrupted: {marker_path}")
        except Exception as e:
            print(f"\n❌ Status: ERROR")
            print(f"   Could not read marker: {e}")
    else:
        print(f"\nStored embeddings: None")
        print(f"✅ Status: CLEAN SLATE")
        print(f"   No existing embeddings found")
    
    # Test provider initialization
    print(f"\nTesting provider initialization...")
    try:
        provider = EmbeddingsProvider()
        print(f"✅ Provider initialized successfully")
        print(f"   Type: {provider.provider_type}")
        print(f"   Model: {provider.model_name}")
        print(f"   Dimension: {provider.get_embedding_dimension()}")
        print(f"   Reranker: {provider.enable_reranker}")
    except Exception as e:
        print(f"❌ Provider initialization failed: {e}")
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())