"""
Test reset utilities - "turn it off and on again" approach.

Forces fresh state between tests by clearing global connection pools
and singleton instances.
"""
import logging
import gc
import asyncio

logger = logging.getLogger(__name__)


def reset_connection_pools():
    """
    Force reset of all connection pools.
    
    Uses the PostgresClient's class method for clean pool reset.
    """
    try:
        # Reset PostgreSQL connection pools using the proper class method
        from clients.postgres_client import PostgresClient
        PostgresClient.reset_all_pools()
    except Exception as e:
        logger.warning(f"Error resetting PostgreSQL pools: {e}")
    
    try:
        # Reset Valkey client and pool
        import clients.valkey_client
        if hasattr(clients.valkey_client, '_valkey_client'):
            clients.valkey_client._valkey_client = None
        if hasattr(clients.valkey_client, '_valkey_pool'):
            clients.valkey_client._valkey_pool = None
        logger.debug("Valkey client reset")
    except Exception as e:
        logger.warning(f"Error resetting Valkey: {e}")


def reset_singletons():
    """
    Reset singleton instances to force fresh initialization.
    """
    try:
        # Reset continuum repository
        import cns.infrastructure.continuum_repository
        if hasattr(cns.infrastructure.continuum_repository, '_continuum_repo_instance'):
            # Clear the _db_cache from the existing instance before resetting
            if cns.infrastructure.continuum_repository._continuum_repo_instance is not None:
                cns.infrastructure.continuum_repository._continuum_repo_instance._db_cache.clear()
            cns.infrastructure.continuum_repository._continuum_repo_instance = None
        logger.debug("Continuum repository singleton reset")
    except Exception as e:
        logger.warning(f"Error resetting continuum repository: {e}")

    try:
        # Reset temporal context service
        import cns.services.temporal_context
        if hasattr(cns.services.temporal_context, '_temporal_service_instance'):
            cns.services.temporal_context._temporal_service_instance = None
        logger.debug("Temporal context singleton reset")
    except Exception as e:
        logger.warning(f"Error resetting temporal context: {e}")

    try:
        # Reset hybrid embeddings provider
        import clients.hybrid_embeddings_provider
        if hasattr(clients.hybrid_embeddings_provider, '_hybrid_embeddings_provider_instance'):
            clients.hybrid_embeddings_provider._hybrid_embeddings_provider_instance = None
        logger.debug("Hybrid embeddings provider singleton reset")
    except Exception as e:
        logger.warning(f"Error resetting hybrid embeddings provider: {e}")


    try:
        # Reset Vault client singleton and cache
        import clients.vault_client
        if hasattr(clients.vault_client, '_vault_client_instance'):
            clients.vault_client._vault_client_instance = None
        if hasattr(clients.vault_client, '_secret_cache'):
            clients.vault_client._secret_cache.clear()
        logger.debug("Vault client singleton and cache reset")
    except Exception as e:
        logger.warning(f"Error resetting Vault client: {e}")


def force_garbage_collection():
    """
    Force garbage collection to clean up any lingering references.
    """
    gc.collect()
    # For async resources, give the event loop a chance to clean up
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(asyncio.sleep(0))
    except RuntimeError:
        pass  # No event loop running


def full_reset():
    """
    Complete reset - the nuclear option.
    
    This is our "turn it off and on again" solution that ensures
    each test starts with completely fresh state.
    """
    logger.info("Performing full test environment reset")
    
    # Step 1: Clear all connection pools
    reset_connection_pools()
    
    # Step 2: Reset all singletons
    reset_singletons()
    
    # Step 3: Force garbage collection
    force_garbage_collection()
    
    logger.info("Full reset complete - environment is fresh")