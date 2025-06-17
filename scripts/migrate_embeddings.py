#!/usr/bin/env python3
"""
Migrate embeddings when switching between providers.

This script handles the migration of embeddings when switching between
local (BGE) and remote (OpenAI) embedding providers. It will:
1. Check the current embedding provider configuration
2. Compare with stored embedding metadata
3. Re-embed all content with the new provider if needed

Usage:
    python scripts/migrate_embeddings.py [--force] [--dry-run]
    
Options:
    --force     Skip confirmation prompt
    --dry-run   Show what would be migrated without making changes
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.embeddings_provider import EmbeddingsProvider, EmbeddingCompatibilityError
from lt_memory.managers.memory_manager import MemoryManager
from tool_relevance_engine import ToolRelevanceEngine
from config import config
from utils.timezone_utils import utc_now

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migrate_embeddings")


def check_migration_needed():
    """Check if migration is needed."""
    marker_path = Path(EmbeddingsProvider.MARKER_FILE)
    
    if not marker_path.exists():
        return None, None, "No existing embeddings found"
    
    try:
        stored = json.loads(marker_path.read_text())
        old_id = f"{stored['provider']}:{stored['model']}"
        
        # Get new provider from current config
        new_provider = config.embeddings.provider
        if new_provider == "local":
            new_model = "BAAI/bge-large-en-v1.5"
        else:
            new_model = config.embeddings.remote.model
        new_id = f"{new_provider}:{new_model}"
        
        return old_id, new_id, None
        
    except Exception as e:
        return None, None, f"Error reading marker file: {e}"


def count_items_to_migrate():
    """Count items that need migration."""
    counts = {
        "memory_blocks": 0,
        "memory_passages": 0,
        "tool_examples": 0,
        "total": 0
    }
    
    try:
        # Count memory items
        memory_manager = MemoryManager()
        with memory_manager.get_session() as session:
            from lt_memory.models import MemoryBlock, MemoryPassage
            counts["memory_blocks"] = session.query(MemoryBlock).count()
            counts["memory_passages"] = session.query(MemoryPassage).count()
    except Exception as e:
        logger.warning(f"Could not count memory items: {e}")
    
    try:
        # Count tool examples
        tools_dir = Path(config.paths.data_dir) / "tools"
        if tools_dir.exists():
            for tool_dir in tools_dir.iterdir():
                if tool_dir.is_dir():
                    examples_file = tool_dir / "classifier_examples.json"
                    if examples_file.exists():
                        examples = json.loads(examples_file.read_text())
                        counts["tool_examples"] += len(examples)
    except Exception as e:
        logger.warning(f"Could not count tool examples: {e}")
    
    counts["total"] = sum(counts.values())
    return counts


def migrate_memory_embeddings(embeddings_provider, dry_run=False):
    """Migrate all memory embeddings."""
    logger.info("Migrating memory embeddings...")
    
    try:
        memory_manager = MemoryManager()
        migrated = 0
        
        with memory_manager.get_session() as session:
            from lt_memory.models import MemoryBlock, MemoryPassage
            
            # Migrate memory blocks
            blocks = session.query(MemoryBlock).all()
            for block in blocks:
                if dry_run:
                    logger.info(f"Would migrate block: {block.name}")
                else:
                    # Re-generate embedding
                    new_embedding = embeddings_provider.encode(block.content)
                    block.embedding = new_embedding.tolist()
                    block.updated_at = utc_now()
                    migrated += 1
                    
                    if migrated % 10 == 0:
                        session.commit()
                        logger.info(f"Migrated {migrated} memory blocks...")
            
            # Migrate memory passages
            passages = session.query(MemoryPassage).all()
            for passage in passages:
                if dry_run:
                    logger.info(f"Would migrate passage from block: {passage.block_id}")
                else:
                    # Re-generate embedding
                    new_embedding = embeddings_provider.encode(passage.content)
                    passage.embedding = new_embedding.tolist()
                    passage.updated_at = utc_now()
                    migrated += 1
                    
                    if migrated % 10 == 0:
                        session.commit()
                        logger.info(f"Migrated {migrated} total items...")
            
            if not dry_run:
                session.commit()
                
        logger.info(f"Migrated {migrated} memory items")
        return migrated
        
    except Exception as e:
        logger.error(f"Error migrating memory: {e}")
        return 0


def migrate_tool_embeddings(embeddings_provider, dry_run=False):
    """Migrate tool example embeddings."""
    logger.info("Migrating tool embeddings...")
    
    try:
        tools_dir = Path(config.paths.data_dir) / "tools"
        migrated = 0
        
        if not tools_dir.exists():
            logger.warning("No tools directory found")
            return 0
        
        for tool_dir in tools_dir.iterdir():
            if not tool_dir.is_dir():
                continue
                
            examples_file = tool_dir / "classifier_examples.json"
            embeddings_file = tool_dir / "classifier_embeddings.npy"
            
            if not examples_file.exists():
                continue
            
            examples = json.loads(examples_file.read_text())
            
            if dry_run:
                logger.info(f"Would migrate {len(examples)} examples for tool: {tool_dir.name}")
                migrated += len(examples)
            else:
                # Re-generate embeddings for all examples
                texts = [ex["query"] for ex in examples]
                new_embeddings = embeddings_provider.encode(texts)
                
                # Save new embeddings
                import numpy as np
                np.save(embeddings_file, new_embeddings)
                
                migrated += len(examples)
                logger.info(f"Migrated {len(examples)} examples for tool: {tool_dir.name}")
        
        logger.info(f"Migrated {migrated} tool examples")
        return migrated
        
    except Exception as e:
        logger.error(f"Error migrating tools: {e}")
        return 0


def main():
    """Main migration function."""
    # Parse arguments
    force = "--force" in sys.argv
    dry_run = "--dry-run" in sys.argv
    
    # Check if migration is needed
    old_id, new_id, error = check_migration_needed()
    
    if error:
        logger.error(error)
        return 1
    
    if not old_id:
        logger.info("No existing embeddings found. Nothing to migrate.")
        return 0
    
    if old_id == new_id:
        logger.info(f"Current config matches stored embeddings ({old_id}). No migration needed.")
        return 0
    
    # Count items to migrate
    counts = count_items_to_migrate()
    
    print("\n" + "="*60)
    print("EMBEDDING MIGRATION REQUIRED")
    print("="*60)
    print(f"From: {old_id}")
    print(f"To:   {new_id}")
    print("\nItems to migrate:")
    print(f"  Memory blocks:    {counts['memory_blocks']:,}")
    print(f"  Memory passages:  {counts['memory_passages']:,}")
    print(f"  Tool examples:    {counts['tool_examples']:,}")
    print(f"  Total:            {counts['total']:,}")
    
    if dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")
    
    print("\n⚠️  WARNING: This operation will:")
    print("  - Re-generate ALL embeddings with the new provider")
    print("  - This can take significant time for large datasets")
    print("  - The system will be unavailable during migration")
    print("="*60)
    
    if not force and not dry_run:
        response = input("\nProceed with migration? (yes/N): ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            return 1
    
    # Delete marker to allow new provider (unless dry run)
    marker_path = Path(EmbeddingsProvider.MARKER_FILE)
    if not dry_run:
        marker_path.unlink()
        logger.info("Removed old provider marker")
    
    try:
        # Initialize new provider
        embeddings = EmbeddingsProvider()
        logger.info(f"Initialized new provider: {new_id}")
        
        # Perform migration
        start_time = datetime.utcnow()
        
        memory_count = migrate_memory_embeddings(embeddings, dry_run)
        tool_count = migrate_tool_embeddings(embeddings, dry_run)
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        
        print("\n" + "="*60)
        if dry_run:
            print("DRY RUN COMPLETE")
        else:
            print("MIGRATION COMPLETE")
        print("="*60)
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Memory items migrated: {memory_count:,}")
        print(f"Tool examples migrated: {tool_count:,}")
        print(f"New provider: {new_id}")
        
        if dry_run:
            print("\nNo changes were made. Remove --dry-run to perform actual migration.")
        else:
            print("\n✅ All embeddings have been migrated successfully!")
        
        return 0
        
    except EmbeddingCompatibilityError as e:
        # This should not happen since we deleted the marker
        logger.error(f"Compatibility error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        
        # Try to restore marker if migration failed
        if not dry_run and not marker_path.exists():
            try:
                marker_data = {
                    "provider": old_id.split(":")[0],
                    "model": old_id.split(":")[1],
                    "dimension": 1024,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "version": "v1"
                }
                marker_path.write_text(json.dumps(marker_data, indent=2))
                logger.info("Restored original provider marker due to migration failure")
            except:
                pass
        
        return 1


if __name__ == "__main__":
    sys.exit(main())