#!/usr/bin/env python3
"""
Setup script for MIRA's ONNX embedding system.

This script downloads and sets up the ONNX embedding model used system-wide
for tool relevance, workflows, and LT_Memory. Also helps initialize the 
LT_Memory database if needed.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def check_postgresql():
    """Check if PostgreSQL is available."""
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ PostgreSQL found: {result.stdout.strip()}")
            return True
        else:
            print("✗ PostgreSQL not found")
            return False
    except FileNotFoundError:
        print("✗ PostgreSQL not found")
        return False

def check_pgvector():
    """Check if pgvector extension is available."""
    db_url = os.getenv("LT_MEMORY_DATABASE_URL")
    if not db_url:
        print("✗ LT_MEMORY_DATABASE_URL not set")
        return False
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                print("✓ pgvector extension found")
                return True
            else:
                print("✗ pgvector extension not installed")
                return False
    except Exception as e:
        print(f"✗ Could not check pgvector: {e}")
        return False

def download_onnx_model():
    """Download and convert ONNX model."""
    model_path = Path("onnx/model.onnx")
    if model_path.exists():
        print("✓ ONNX model already exists")
        return True
    
    print("Downloading and converting ONNX model...")
    try:
        # Create onnx directory
        model_path.parent.mkdir(exist_ok=True)
        
        # Use optimum CLI for ONNX export
        result = subprocess.run([
            sys.executable, "-m", "optimum.exporters.onnx",
            "--model", "sentence-transformers/all-MiniLM-L6-v2",
            "--task", "feature-extraction",
            "onnx/"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ ONNX model downloaded successfully")
            return True
        else:
            print(f"✗ Failed to download ONNX model: {result.stderr}")
            # Try alternative method
            print("Trying alternative export method...")
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            
            # Load and save the model
            model = ORTModelForFeatureExtraction.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                export=True
            )
            model.save_pretrained("onnx/")
            
            print("✓ ONNX model downloaded successfully (alternative method)")
            return True
    except Exception as e:
        print(f"✗ Error downloading ONNX model: {e}")
        return False

def initialize_database():
    """Initialize the LT_Memory database."""
    print("Initializing LT_Memory database...")
    try:
        from lt_memory.migrations.init_database import init_database
        
        db_url = os.getenv("LT_MEMORY_DATABASE_URL")
        if not db_url:
            print("✗ LT_MEMORY_DATABASE_URL not set")
            return False
        
        init_database(db_url)
        print("✓ Database initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False

def main():
    """Main setup function."""
    print("MIRA ONNX Embedding & LT_Memory Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    print("\n1. Checking prerequisites...")
    postgres_ok = check_postgresql()
    pgvector_ok = check_pgvector()
    
    if not postgres_ok:
        print("\nPlease install PostgreSQL:")
        print("  Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
        print("  macOS: brew install postgresql")
        return False
    
    if not pgvector_ok:
        print("\nPlease install pgvector extension:")
        print("  Ubuntu/Debian: sudo apt-get install postgresql-14-pgvector")
        print("  macOS: brew install pgvector")
        print("  Then connect to your database and run: CREATE EXTENSION vector;")
        return False
    
    # Download ONNX model
    print("\n2. Setting up ONNX model...")
    if not download_onnx_model():
        print("Failed to download ONNX model")
        return False
    
    # Initialize database
    print("\n3. Initializing database...")
    if not initialize_database():
        print("Failed to initialize database")
        return False
    
    print("\n✅ MIRA ONNX embedding and LT_Memory setup completed successfully!")
    print("\nNext steps:")
    print("1. Set LT_MEMORY_DATABASE_URL environment variable")
    print("2. Run MIRA with: python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)