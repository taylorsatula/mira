"""
Currency conversion tool that demonstrates advanced tool implementation patterns.

This tool provides currency conversion with real-time and historical rates.
It demonstrates the provider pattern, caching, validation, and other advanced
patterns for building maintainable, extensible tools.
"""

# Standard library imports
import logging
import os
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Third-party imports
import requests
from pydantic import BaseModel, Field

# Local imports
from tools.repo import Tool
from errors import ErrorCode, error_context, ToolError
from config.registry import registry


# -------------------- CONFIGURATION --------------------
# Complex tools often use nested configuration models to organize related settings

class CurrencyProvider(BaseModel):
    """
    Configuration for a currency data provider.
    
    This nested model defines the configuration for a single exchange rate provider.
    Complex tools can use nested models to organize related configuration parameters.
    """
    name: str = Field(description="Provider name")
    api_key: str = Field(default="", description="API key for the provider")
    url: str = Field(description="Provider API URL")
    priority: int = Field(default=1, description="Priority order (lower is higher priority)")
    enabled: bool = Field(default=True, description="Whether this provider is enabled")


class CurrencyToolConfig(BaseModel):
    """
    Configuration for the currency_tool.
    
    This demonstrates a more complex configuration structure with nested models
    and multiple configuration sections.
    """
    # Standard configuration - all tools should have this
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")
    
    # Provider configuration - using nested models for related settings
    providers: Dict[str, CurrencyProvider] = Field(
        default={
            "exchangerate-api": CurrencyProvider(
                name="ExchangeRate-API",
                url="https://api.exchangerate-api.com/v4/latest/",
                priority=1
            ),
            "openexchangerates": CurrencyProvider(
                name="Open Exchange Rates",
                url="https://openexchangerates.org/api/",
                priority=2
            )
        },
        description="Available currency data providers"
    )
    
    # Default values
    default_base_currency: str = Field(default="USD", description="Default base currency")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, description="Whether to cache exchange rates")
    cache_duration: int = Field(default=3600, description="Cache duration in seconds (default: 1 hour)")
    cache_directory: str = Field(default="data/currency_tool/cache", description="Directory to store cached rates")
    
    # Functional settings
    decimal_places: int = Field(default=2, description="Number of decimal places in results")
    max_historical_days: int = Field(default=365, description="Maximum days in the past for historical rates")

# Register with registry
registry.register("currency_tool", CurrencyToolConfig)


# -------------------- VALIDATION UTILITIES --------------------
# Complex tools use a validation class to centralize validation logic

class ValidationUtils:
    """
    Utility methods for validating currency tool parameters.
    
    For complex tools, using a class with static methods to centralize
    validation logic improves maintainability and encourages reuse.
    """
    
    @staticmethod
    def validate_currency_code(code: Optional[str], param_name: str) -> str:
        """
        Validate a currency code.
        
        Args:
            code: Currency code to validate
            param_name: Parameter name for error messages
            
        Returns:
            Validated currency code (normalized to uppercase)
            
        Raises:
            ToolError: If currency code is invalid
        """
        # Check if parameter is missing or not a string
        if not code or not isinstance(code, str):
            raise ToolError(
                f"{param_name} must be a non-empty string",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: code}
            )
        
        # Convert to uppercase and check format
        code = code.upper()
        if len(code) != 3:
            raise ToolError(
                f"{param_name} must be a 3-letter currency code (e.g., USD, EUR)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: code}
            )
            
        # Return the validated and normalized code
        return code
    
    @staticmethod
    def validate_amount(amount: Any, param_name: str = "amount") -> float:
        """
        Validate an amount value.
        
        Args:
            amount: Amount to validate
            param_name: Parameter name for error messages
            
        Returns:
            Validated amount as float
            
        Raises:
            ToolError: If amount is invalid
        """
        # Check if parameter is missing
        if amount is None:
            raise ToolError(
                f"{param_name} is required and must be a number",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: amount}
            )
            
        # Try to convert to float
        try:
            amount_float = float(amount)
        except (ValueError, TypeError):
            raise ToolError(
                f"{param_name} must be a valid number, got {type(amount).__name__}: {amount}",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: amount}
            )
            
        # Check for positive value
        if amount_float <= 0:
            raise ToolError(
                f"{param_name} must be greater than zero",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: amount_float}
            )
            
        return amount_float
    
    @staticmethod
    def validate_date(date_str: Optional[str], param_name: str = "date") -> Optional[datetime]:
        """
        Validate and parse a date string.
        
        Args:
            date_str: Date string in ISO format (YYYY-MM-DD)
            param_name: Parameter name for error messages
            
        Returns:
            Parsed datetime object or None if date_str is None
            
        Raises:
            ToolError: If date is invalid
        """
        # Allow None value
        if date_str is None:
            return None
            
        # Check if parameter is a string
        if not isinstance(date_str, str):
            raise ToolError(
                f"{param_name} must be a string in ISO format (YYYY-MM-DD)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: date_str}
            )
            
        try:
            # Parse date in ISO format
            date_obj = datetime.fromisoformat(date_str)
            
            # Ensure date is not in the future
            if date_obj.date() > datetime.now().date():
                raise ToolError(
                    f"{param_name} cannot be in the future",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {param_name: date_str}
                )
                
            return date_obj
        except ValueError:
            raise ToolError(
                f"Invalid {param_name} format: '{date_str}'. Use ISO format (YYYY-MM-DD)",
                ErrorCode.TOOL_INVALID_INPUT,
                {param_name: date_str}
            )


# -------------------- CACHING SYSTEM --------------------
# Complex tools often implement caching to optimize performance

class RateCache:
    """
    Manages caching of currency exchange rates.
    
    This component encapsulates all caching logic, keeping the main tool
    class focused on its core responsibilities (separation of concerns).
    """
    
    def __init__(self, cache_dir: str, cache_duration: int):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            cache_duration: Cache validity duration in seconds
        """
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str) -> str:
        """
        Get the cache file path for a given key.
        
        Args:
            key: Cache key (e.g., USD_2023-05-15)
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the key for the filename to avoid invalid characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def is_valid(self, cache_path: str) -> bool:
        """
        Check if a cache file is valid and not expired.
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(cache_path):
            return False
            
        # Check if cache is expired based on file modification time
        cache_age = time.time() - os.path.getmtime(cache_path)
        return cache_age < self.cache_duration
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if available and valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not available
        """
        cache_path = self.get_cache_path(key)
        
        # Check if cache is valid
        if not self.is_valid(cache_path):
            return None
            
        # Read from cache
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache for {key}: {str(e)}")
            return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_path = self.get_cache_path(key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to cache data for {key}: {str(e)}")


# -------------------- PROVIDER ARCHITECTURE --------------------
# The provider pattern abstracts different implementations behind a common interface

class ExchangeRateProvider(ABC):
    """
    Abstract base class for exchange rate providers.
    
    This abstract class defines the interface that all exchange rate providers
    must implement, allowing the tool to work with different providers interchangeably.
    """
    
    def __init__(self, api_key: str, base_url: str, logger: logging.Logger):
        """
        Initialize the exchange rate provider.
        
        Args:
            api_key: API key for the provider
            base_url: Base URL for API requests
            logger: Logger instance
        """
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger
    
    @abstractmethod
    def get_latest_rates(self, base_currency: str) -> Dict[str, float]:
        """
        Get latest exchange rates for a base currency.
        
        Args:
            base_currency: Base currency code
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_historical_rates(self, base_currency: str, date: datetime) -> Dict[str, float]:
        """
        Get historical exchange rates for a base currency.
        
        Args:
            base_currency: Base currency code
            date: Historical date
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        pass


class ExchangeRateAPIProvider(ExchangeRateProvider):
    """
    Exchange rate provider implementation using ExchangeRate-API.
    
    This concrete provider implements the ExchangeRateProvider interface
    for the ExchangeRate-API service.
    """
    
    def get_latest_rates(self, base_currency: str) -> Dict[str, float]:
        """
        Get latest exchange rates from ExchangeRate-API.
        
        Args:
            base_currency: Base currency code
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        try:
            # Build API URL
            url = f"{self.base_url}{base_currency}"
            
            # Add API key if provided
            params = {}
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Make API request with proper error handling
            self.logger.info(f"Fetching latest rates for {base_currency} from ExchangeRate-API")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()  # Raises an exception for HTTP errors
            data = response.json()
            
            # Validate the response format
            if "rates" not in data:
                raise ToolError(
                    "Invalid response format from exchange rate provider",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"response": data}
                )
                
            return data["rates"]
        except requests.RequestException as e:
            # Handle network errors specifically
            raise ToolError(
                f"Failed to fetch exchange rates: {str(e)}",
                ErrorCode.API_CONNECTION_ERROR,
                {"base_currency": base_currency, "error": str(e)}
            )
        except (ValueError, KeyError) as e:
            # Handle JSON parsing errors and missing keys
            raise ToolError(
                f"Invalid response from exchange rate provider: {str(e)}",
                ErrorCode.API_RESPONSE_ERROR,
                {"base_currency": base_currency, "error": str(e)}
            )
    
    def get_historical_rates(self, base_currency: str, date: datetime) -> Dict[str, float]:
        """
        Get historical exchange rates from ExchangeRate-API.
        
        Args:
            base_currency: Base currency code
            date: Historical date
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        try:
            # Format date for API
            date_str = date.strftime("%Y-%m-%d")
            
            # Build API URL - the API structure varies by provider
            url = f"{self.base_url}history"
            
            # Add API key and parameters
            params = {
                "base": base_currency,
                "date": date_str
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            # Make API request
            self.logger.info(f"Fetching historical rates for {base_currency} on {date_str}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract rates - format depends on provider
            if "rates" not in data:
                raise ToolError(
                    "Invalid response format from exchange rate provider",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"response": data}
                )
                
            return data["rates"]
        except requests.RequestException as e:
            raise ToolError(
                f"Failed to fetch historical exchange rates: {str(e)}",
                ErrorCode.API_CONNECTION_ERROR,
                {"base_currency": base_currency, "date": date.isoformat(), "error": str(e)}
            )
        except (ValueError, KeyError) as e:
            raise ToolError(
                f"Invalid response from exchange rate provider: {str(e)}",
                ErrorCode.API_RESPONSE_ERROR,
                {"base_currency": base_currency, "date": date.isoformat(), "error": str(e)}
            )


class OpenExchangeRatesProvider(ExchangeRateProvider):
    """
    Exchange rate provider implementation using Open Exchange Rates.
    
    This second provider shows how the provider pattern allows you to
    implement multiple backends for the same functionality.
    """
    
    def get_latest_rates(self, base_currency: str) -> Dict[str, float]:
        """
        Get latest exchange rates from Open Exchange Rates.
        
        Args:
            base_currency: Base currency code
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        try:
            # Build API URL - this API always uses USD as base
            url = f"{self.base_url}latest.json"
            
            # Add API key and parameters
            params = {
                "app_id": self.api_key,
                "base": "USD"  # Base is always USD for this API (unless premium)
            }
            
            # Make API request
            self.logger.info(f"Fetching latest rates from Open Exchange Rates")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract rates
            if "rates" not in data:
                raise ToolError(
                    "Invalid response format from exchange rate provider",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"response": data}
                )
                
            rates = data["rates"]
            
            # If requested base is not USD, convert rates
            # This demonstrates handling provider-specific limitations
            if base_currency != "USD":
                # Get the rate for the requested base
                if base_currency not in rates:
                    raise ToolError(
                        f"Base currency {base_currency} not found in available rates",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"base_currency": base_currency, "available_currencies": list(rates.keys())}
                    )
                
                base_rate = rates[base_currency]
                
                # Convert all rates to the requested base
                converted_rates = {}
                for currency, rate in rates.items():
                    converted_rates[currency] = rate / base_rate
                
                return converted_rates
            else:
                return rates
        except requests.RequestException as e:
            raise ToolError(
                f"Failed to fetch exchange rates: {str(e)}",
                ErrorCode.API_CONNECTION_ERROR,
                {"base_currency": base_currency, "error": str(e)}
            )
        except (ValueError, KeyError) as e:
            raise ToolError(
                f"Invalid response from exchange rate provider: {str(e)}",
                ErrorCode.API_RESPONSE_ERROR,
                {"base_currency": base_currency, "error": str(e)}
            )
    
    def get_historical_rates(self, base_currency: str, date: datetime) -> Dict[str, float]:
        """
        Get historical exchange rates from Open Exchange Rates.
        
        Args:
            base_currency: Base currency code
            date: Historical date
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        try:
            # Format date for API
            date_str = date.strftime("%Y-%m-%d")
            
            # Build API URL for historical data
            url = f"{self.base_url}historical/{date_str}.json"
            
            # Add API key and parameters
            params = {
                "app_id": self.api_key,
                "base": "USD"  # Base is always USD for this API (unless premium)
            }
            
            # Make API request
            self.logger.info(f"Fetching historical rates for {date_str} from Open Exchange Rates")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract rates
            if "rates" not in data:
                raise ToolError(
                    "Invalid response format from exchange rate provider",
                    ErrorCode.API_RESPONSE_ERROR,
                    {"response": data}
                )
                
            rates = data["rates"]
            
            # If requested base is not USD, convert rates
            if base_currency != "USD":
                # Get the rate for the requested base
                if base_currency not in rates:
                    raise ToolError(
                        f"Base currency {base_currency} not found in available rates",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"base_currency": base_currency, "available_currencies": list(rates.keys())}
                    )
                
                base_rate = rates[base_currency]
                
                # Convert all rates to the requested base
                converted_rates = {}
                for currency, rate in rates.items():
                    converted_rates[currency] = rate / base_rate
                
                return converted_rates
            else:
                return rates
        except requests.RequestException as e:
            raise ToolError(
                f"Failed to fetch historical exchange rates: {str(e)}",
                ErrorCode.API_CONNECTION_ERROR,
                {"base_currency": base_currency, "date": date.isoformat(), "error": str(e)}
            )
        except (ValueError, KeyError) as e:
            raise ToolError(
                f"Invalid response from exchange rate provider: {str(e)}",
                ErrorCode.API_RESPONSE_ERROR,
                {"base_currency": base_currency, "date": date.isoformat(), "error": str(e)}
            )


# -------------------- MAIN TOOL CLASS --------------------

class CurrencyTool(Tool):
    """
    Tool for currency conversion with real-time and historical rates.
    
    This tool provides currency conversion between different currencies
    using current or historical exchange rates. It supports multiple
    exchange rate data providers and includes caching for efficiency.
    
    This is a complex tool example demonstrating:
    1. Provider pattern for multiple implementations
    2. Caching for performance optimization
    3. Validation utilities
    4. Component-based architecture
    5. Sophisticated error handling with fallbacks
    """
    
    name = "currency_tool"
    
    description = """
    Converts amounts between different currencies using real-time or historical exchange rates.
    Use this tool when the user wants to convert money from one currency to another or get
    exchange rate information.
    
    OPERATIONS:
    - convert: Converts an amount from one currency to another
      Parameters:
        amount (required): Amount to convert
        from_currency (required): Source currency code (e.g., USD, EUR)
        to_currency (required): Target currency code (e.g., JPY, GBP)
        date (optional): Date for historical conversion in ISO format (YYYY-MM-DD). If not provided, uses current rates.
    
    - get_rate: Gets the exchange rate between two currencies
      Parameters:
        from_currency (required): Source currency code (e.g., USD, EUR)
        to_currency (required): Target currency code (e.g., JPY, GBP)
        date (optional): Date for historical rate in ISO format (YYYY-MM-DD). If not provided, uses current rates.
    
    - list_currencies: Lists available currencies
      Parameters:
        None
    
    RESPONSE FORMAT:
    - For convert operations: Original amount, converted amount, exchange rate used
    - For get_rate operations: Exchange rate between the two currencies
    - For list_currencies operations: List of available currency codes
    
    LIMITATIONS:
    - Some currency pairs may not be directly available and will use USD as an intermediate
    - Historical rates are limited to the past year
    - Exchange rates are typically updated once per day
    """
    
    usage_examples = [
        {
            "input": {
                "operation": "convert",
                "amount": 100,
                "from_currency": "USD",
                "to_currency": "EUR"
            },
            "output": {
                "success": True,
                "original": {
                    "amount": 100,
                    "currency": "USD"
                },
                "converted": {
                    "amount": 85.23,
                    "currency": "EUR"
                },
                "rate": 0.8523,
                "date": "2023-09-15"
            }
        },
        {
            "input": {
                "operation": "get_rate",
                "from_currency": "USD",
                "to_currency": "JPY",
                "date": "2023-01-15"
            },
            "output": {
                "success": True,
                "from_currency": "USD",
                "to_currency": "JPY",
                "rate": 128.43,
                "date": "2023-01-15"
            }
        }
    ]
    
    def __init__(self):
        """Initialize the currency tool."""
        super().__init__()
        self.logger.info("CurrencyTool initialized")
        
        # Create required directories early, but delay provider creation
        # until needed to ensure we have the latest configuration
        from config import config
        os.makedirs(config.currency_tool.cache_directory, exist_ok=True)
    
    # -------------------- COMPONENT FACTORY METHODS --------------------
    # These methods create and configure the tool's components
    
    def _create_rate_cache(self) -> RateCache:
        """
        Create and configure a rate cache instance.
        
        Returns:
            Configured rate cache
        """
        # Import config when needed to avoid circular imports
        from config import config
        
        cache_dir = config.currency_tool.cache_directory
        cache_duration = config.currency_tool.cache_duration
        
        self.logger.debug(f"Creating cache with directory={cache_dir}, duration={cache_duration}s")
        return RateCache(cache_dir, cache_duration)
    
    def _create_providers(self) -> List[ExchangeRateProvider]:
        """
        Create and configure exchange rate providers.
        
        Returns:
            List of configured exchange rate providers ordered by priority
        """
        # Import config when needed
        from config import config
        
        providers = []
        
        # Loop through configured providers and create instances
        for provider_id, provider_config in config.currency_tool.providers.items():
            # Skip disabled providers
            if not provider_config.enabled:
                self.logger.debug(f"Provider {provider_id} is disabled, skipping")
                continue
            
            # Create the appropriate provider based on the provider type
            self.logger.debug(f"Creating provider {provider_id} with priority {provider_config.priority}")
            if provider_id == "exchangerate-api":
                providers.append(
                    ExchangeRateAPIProvider(
                        api_key=provider_config.api_key,
                        base_url=provider_config.url,
                        logger=self.logger
                    )
                )
            elif provider_id == "openexchangerates":
                providers.append(
                    OpenExchangeRatesProvider(
                        api_key=provider_config.api_key,
                        base_url=provider_config.url,
                        logger=self.logger
                    )
                )
            else:
                self.logger.warning(f"Unknown provider type: {provider_id}")
        
        # Sort by priority (lower number = higher priority)
        provider_list = sorted(
            [(p, config.currency_tool.providers[pid].priority) 
             for pid, p in enumerate(providers)],
            key=lambda x: x[1]
        )
        
        # Return just the provider instances
        return [p for p, _ in provider_list]
    
    # -------------------- CORE FUNCTIONALITY METHODS --------------------
    # These methods implement the tool's core functionality
    
    def _get_exchange_rates(
        self,
        base_currency: str,
        date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get exchange rates for a base currency, with caching.
        
        This method demonstrates:
        1. Caching for performance
        2. Provider fallback for reliability
        3. Error propagation
        
        Args:
            base_currency: Base currency code
            date: Optional date for historical rates
            
        Returns:
            Dictionary of currency codes to exchange rates
            
        Raises:
            ToolError: If rates cannot be retrieved
        """
        # Import config
        from config import config
        
        # Create cache if enabled
        cache = self._create_rate_cache() if config.currency_tool.cache_enabled else None
        
        # Generate cache key if caching is enabled
        cache_key = None
        if cache:
            date_str = date.strftime("%Y-%m-%d") if date else "latest"
            cache_key = f"{base_currency}_{date_str}"
            
            # Try to get from cache first
            cached_data = cache.get(cache_key)
            if cached_data:
                self.logger.info(f"Using cached rates for {base_currency} on {date_str}")
                return cached_data
        
        # Create providers
        providers = self._create_providers()
        
        if not providers:
            raise ToolError(
                "No exchange rate providers available",
                ErrorCode.TOOL_CONFIGURATION_ERROR,
                {}
            )
        
        # Try each provider in order of priority - this is the fallback mechanism
        self.logger.info(f"Trying {len(providers)} providers to get {'historical' if date else 'latest'} rates")
        last_error = None
        for i, provider in enumerate(providers):
            try:
                # Get rates from the provider based on whether we need historical or current rates
                if date:
                    self.logger.debug(f"Trying provider {i+1}/{len(providers)} for historical rates")
                    rates = provider.get_historical_rates(base_currency, date)
                else:
                    self.logger.debug(f"Trying provider {i+1}/{len(providers)} for latest rates")
                    rates = provider.get_latest_rates(base_currency)
                
                # Cache the result if caching is enabled
                if cache and cache_key:
                    self.logger.debug(f"Caching rates with key {cache_key}")
                    cache.set(cache_key, rates)
                
                return rates
            except Exception as e:
                # Log the error and try the next provider
                self.logger.warning(f"Provider {i+1}/{len(providers)} failed: {str(e)}")
                last_error = e
        
        # If we get here, all providers failed
        if last_error:
            # Propagate ToolError directly, convert other exceptions to ToolError
            if isinstance(last_error, ToolError):
                raise last_error
            else:
                raise ToolError(
                    f"All exchange rate providers failed: {str(last_error)}",
                    ErrorCode.TOOL_EXECUTION_ERROR,
                    {"error": str(last_error)}
                )
        else:
            raise ToolError(
                "No exchange rate providers available",
                ErrorCode.TOOL_CONFIGURATION_ERROR,
                {}
            )
    
    def _convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Convert an amount between currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code
            date: Optional date for historical conversion
            
        Returns:
            Dictionary with conversion details
            
        Raises:
            ToolError: If conversion fails
        """
        self.logger.info(f"Converting {amount} {from_currency} to {to_currency}")
        
        # Get exchange rates for the source currency
        rates = self._get_exchange_rates(from_currency, date)
        
        # Check if target currency is available
        if to_currency not in rates:
            raise ToolError(
                f"Target currency {to_currency} not found in available rates",
                ErrorCode.TOOL_INVALID_INPUT,
                {"to_currency": to_currency, "available_currencies": list(rates.keys())}
            )
        
        # Get the exchange rate
        rate = rates[to_currency]
        
        # Calculate converted amount
        converted_amount = amount * rate
        
        # Round to configured decimal places
        from config import config
        decimal_places = config.currency_tool.decimal_places
        converted_amount = round(converted_amount, decimal_places)
        
        # Format date for response
        date_str = date.strftime("%Y-%m-%d") if date else datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Converted {amount} {from_currency} to {converted_amount} {to_currency} at rate {rate}")
        
        # Return structured result
        return {
            "original": {
                "amount": amount,
                "currency": from_currency
            },
            "converted": {
                "amount": converted_amount,
                "currency": to_currency
            },
            "rate": rate,
            "date": date_str
        }
    
    def _get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get the exchange rate between two currencies.
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            date: Optional date for historical rate
            
        Returns:
            Dictionary with exchange rate details
            
        Raises:
            ToolError: If rate cannot be retrieved
        """
        self.logger.info(f"Getting exchange rate from {from_currency} to {to_currency}")
        
        # Get exchange rates for the source currency
        rates = self._get_exchange_rates(from_currency, date)
        
        # Check if target currency is available
        if to_currency not in rates:
            raise ToolError(
                f"Target currency {to_currency} not found in available rates",
                ErrorCode.TOOL_INVALID_INPUT,
                {"to_currency": to_currency, "available_currencies": list(rates.keys())}
            )
        
        # Get the exchange rate
        rate = rates[to_currency]
        
        # Round to configured decimal places
        from config import config
        decimal_places = config.currency_tool.decimal_places
        rate = round(rate, decimal_places)
        
        # Format date for response
        date_str = date.strftime("%Y-%m-%d") if date else datetime.now().strftime("%Y-%m-%d")
        
        return {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": rate,
            "date": date_str
        }
    
    def _list_currencies(self) -> Dict[str, Any]:
        """
        List available currencies.
        
        Returns:
            Dictionary with list of available currencies
            
        Raises:
            ToolError: If currencies cannot be listed
        """
        self.logger.info("Listing available currencies")
        
        # Get exchange rates for a major currency (USD)
        # to see which currencies are available
        rates = self._get_exchange_rates("USD")
        
        # Add the base currency to the list
        currencies = ["USD"] + list(rates.keys())
        
        # Sort alphabetically
        currencies.sort()
        
        self.logger.info(f"Found {len(currencies)} available currencies")
        
        return {
            "currencies": currencies,
            "count": len(currencies)
        }
    
    # -------------------- MAIN ENTRY POINT --------------------
    
    def run(
        self,
        operation: str,
        amount: Optional[Union[int, float]] = None,
        from_currency: Optional[str] = None,
        to_currency: Optional[str] = None,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the currency tool with the specified operation.
        
        This is the main entry point for the tool, demonstrating:
        1. Parameter validation
        2. Operation routing
        3. Error handling
        4. Standardized responses
        
        Args:
            operation: The operation to perform (convert, get_rate, list_currencies)
            amount: Amount to convert (for convert operation)
            from_currency: Source currency code
            to_currency: Target currency code
            date: Optional date for historical rates (ISO format: YYYY-MM-DD)
            
        Returns:
            Dictionary containing the operation results with standardized format:
            {
                "success": True,              # Whether the operation succeeded
                ... operation-specific data ...
            }
            
        Raises:
            ToolError: If the operation fails or parameters are invalid
        """
        self.logger.info(f"Running currency tool with operation: {operation}")
        
        # Use error_context to wrap all operations for consistent error handling
        with error_context(
            component_name=self.name,
            operation=f"executing currency operation '{operation}'",
            error_class=ToolError,
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            logger=self.logger
        ):
            # Import config
            from config import config
            
            # Validate operation first - this determines what other parameters are needed
            if operation not in ["convert", "get_rate", "list_currencies"]:
                raise ToolError(
                    f"Invalid operation: {operation}. Must be one of: convert, get_rate, list_currencies",
                    ErrorCode.TOOL_INVALID_INPUT,
                    {"provided_operation": operation}
                )
            
            # Parse date if provided
            date_obj = ValidationUtils.validate_date(date) if date else None
            
            # Check date limits if a date was provided
            if date_obj:
                max_days = config.currency_tool.max_historical_days
                oldest_allowed = datetime.now() - timedelta(days=max_days)
                
                if date_obj < oldest_allowed:
                    raise ToolError(
                        f"Historical rates are only available for the past {max_days} days",
                        ErrorCode.TOOL_INVALID_INPUT,
                        {"date": date, "max_days": max_days}
                    )
            
            # Execute the requested operation
            if operation == "convert":
                # For convert operation, validate required parameters
                amount_float = ValidationUtils.validate_amount(amount)
                from_curr = ValidationUtils.validate_currency_code(from_currency, "from_currency")
                to_curr = ValidationUtils.validate_currency_code(to_currency, "to_currency")
                
                # Perform the conversion
                result = self._convert_currency(amount_float, from_curr, to_curr, date_obj)
                
                # Return standardized response format
                return {
                    "success": True,
                    **result
                }
                
            elif operation == "get_rate":
                # For get_rate operation, validate required parameters
                from_curr = ValidationUtils.validate_currency_code(from_currency, "from_currency")
                to_curr = ValidationUtils.validate_currency_code(to_currency, "to_currency")
                
                # Get the exchange rate
                result = self._get_exchange_rate(from_curr, to_curr, date_obj)
                
                # Return standardized response format
                return {
                    "success": True,
                    **result
                }
                
            else:  # list_currencies
                # List_currencies operation doesn't require additional parameters
                result = self._list_currencies()
                
                # Return standardized response format
                return {
                    "success": True,
                    **result
                }