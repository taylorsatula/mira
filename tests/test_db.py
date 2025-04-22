"""
Unit tests for the database module.
"""

import os
import unittest
import tempfile
from datetime import datetime
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base, Customer, Database
from config import config


class TestDatabaseModule(unittest.TestCase):
    """Test cases for the database module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary database file
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db")
        
        # Override database URI in config
        self.original_uri = config.database.uri
        config.database.uri = f"sqlite:///{self.temp_file.name}"
        
        # Create tables directly
        engine = create_engine(config.database.uri)
        Base.metadata.create_all(engine)
        self.Session = sessionmaker(bind=engine)
        
        # Reset Database singleton for testing
        Database._instance = None
        Database._engine = None
        Database._session_factory = None
        
        # Create a test database instance
        self.db = Database()
        
    def tearDown(self):
        """Clean up test environment."""
        # Reset config
        config.database.uri = self.original_uri
        
        # Close and delete temporary file
        self.temp_file.close()
        
        # Reset Database singleton
        Database._instance = None
        Database._engine = None
        Database._session_factory = None
    
    def test_customer_model_conversion(self):
        """Test conversion between Customer model and dictionary."""
        # Create a customer dictionary
        customer_dict = {
            "id": "test123",
            "given_name": "John",
            "family_name": "Doe",
            "email_address": "john.doe@example.com",
            "phone_number": "555-1234",
            "address": {
                "address_line_1": "123 Main St",
                "address_line_2": "Apt 4B",
                "locality": "Anytown",
                "administrative_district_level_1": "CA",
                "postal_code": "12345",
                "country": "US"
            },
            "geocoding_data": {
                "coordinates": {
                    "lat": 37.7749,
                    "lng": -122.4194
                },
                "geocoded_at": 1617290400
            },
            "created_at": "2023-01-01T12:00:00",
            "updated_at": "2023-01-02T12:00:00",
            "custom_field": "Custom value"
        }
        
        # Convert to model
        customer = Customer.from_dict(customer_dict)
        
        # Check fields
        self.assertEqual(customer.id, "test123")
        self.assertEqual(customer.given_name, "John")
        self.assertEqual(customer.family_name, "Doe")
        self.assertEqual(customer.email_address, "john.doe@example.com")
        self.assertEqual(customer.phone_number, "555-1234")
        self.assertEqual(customer.address_line1, "123 Main St")
        self.assertEqual(customer.address_line2, "Apt 4B")
        self.assertEqual(customer.city, "Anytown")
        self.assertEqual(customer.state, "CA")
        self.assertEqual(customer.postal_code, "12345")
        self.assertEqual(customer.country, "US")
        self.assertAlmostEqual(customer.latitude, 37.7749)
        self.assertAlmostEqual(customer.longitude, -122.4194)
        self.assertEqual(customer.additional_data, {"custom_field": "Custom value"})
        
        # Convert back to dictionary
        result_dict = customer.to_dict()
        
        # Check main fields
        self.assertEqual(result_dict["id"], "test123")
        self.assertEqual(result_dict["given_name"], "John")
        self.assertEqual(result_dict["family_name"], "Doe")
        self.assertEqual(result_dict["email_address"], "john.doe@example.com")
        self.assertEqual(result_dict["phone_number"], "555-1234")
        
        # Check address structure
        self.assertIn("address", result_dict)
        self.assertEqual(result_dict["address"]["address_line_1"], "123 Main St")
        self.assertEqual(result_dict["address"]["address_line_2"], "Apt 4B")
        
        # Check geocoding structure
        self.assertIn("geocoding_data", result_dict)
        self.assertAlmostEqual(result_dict["geocoding_data"]["coordinates"]["lat"], 37.7749)
        self.assertAlmostEqual(result_dict["geocoding_data"]["coordinates"]["lng"], -122.4194)
        
        # Check additional fields
        self.assertEqual(result_dict["custom_field"], "Custom value")
    
    def test_database_crud_operations(self):
        """Test CRUD operations with the database interface."""
        # Create a test customer
        customer = Customer(
            id="crud_test",
            given_name="Jane",
            family_name="Smith",
            email_address="jane.smith@example.com",
            latitude=34.0522,
            longitude=-118.2437
        )
        
        # Test add
        added = self.db.add(customer)
        self.assertEqual(added.id, "crud_test")
        
        # Test get
        retrieved = self.db.get(Customer, "crud_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.email_address, "jane.smith@example.com")
        
        # Test query
        from sqlalchemy import func
        results = self.db.query(Customer, func.lower(Customer.given_name).like("%jane%"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "crud_test")
        
        # Test update
        retrieved.phone_number = "555-5678"
        updated = self.db.update(retrieved)
        self.assertEqual(updated.phone_number, "555-5678")
        
        # Verify update persisted
        re_retrieved = self.db.get(Customer, "crud_test")
        self.assertEqual(re_retrieved.phone_number, "555-5678")
        
        # Test delete
        result = self.db.delete(re_retrieved)
        self.assertTrue(result)
        
        # Verify deletion
        deleted_check = self.db.get(Customer, "crud_test")
        self.assertIsNone(deleted_check)
    
    def test_execute_raw_sql(self):
        """Test executing raw SQL statements."""
        # Add test data
        customer1 = Customer(id="sql_test1", given_name="Alice", family_name="Johnson")
        customer2 = Customer(id="sql_test2", given_name="Bob", family_name="Williams")
        
        self.db.add(customer1)
        self.db.add(customer2)
        
        # Execute raw SQL query
        result = self.db.execute("SELECT id, given_name FROM customers WHERE id LIKE 'sql_test%'")
        rows = list(result)
        
        # Check results
        self.assertEqual(len(rows), 2)
        
        # Check that the rows contain the expected data
        ids = [row[0] for row in rows]
        self.assertIn("sql_test1", ids)
        self.assertIn("sql_test2", ids)


if __name__ == "__main__":
    unittest.main()