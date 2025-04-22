"""
Unit tests for the database-backed customer tool.
"""

import os
import json
import unittest
import tempfile
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db import Base, Customer, Database
from tools.customer_tool_db import CustomerToolDB
from config import config
from errors import ToolError


class TestCustomerToolDB(unittest.TestCase):
    """Test cases for the customer_db_tool with database support."""
    
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
        
        # Reset Database singleton for testing
        Database._instance = None
        Database._engine = None
        Database._session_factory = None
        
        # Create test data
        self.db = Database()
        self._create_test_data()
        
        # Create a test instance of CustomerToolDB
        self.tool = CustomerToolDB()
    
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
    
    def _create_test_data(self):
        """Create test customer data."""
        # Add test customers to database
        customers = [
            Customer(
                id="cust1",
                given_name="John",
                family_name="Smith",
                email_address="john.smith@example.com",
                phone_number="555-1234",
                address_line1="123 Main St",
                city="Anytown",
                state="CA",
                postal_code="12345",
                country="US",
                latitude=37.7749,
                longitude=-122.4194
            ),
            Customer(
                id="cust2",
                given_name="Jane",
                family_name="Doe",
                email_address="jane.doe@example.com",
                phone_number="555-5678",
                address_line1="456 Oak Ave",
                city="Somecity",
                state="NY",
                postal_code="54321",
                country="US",
                latitude=40.7128,
                longitude=-74.0060
            ),
            Customer(
                id="cust3",
                given_name="Robert",
                family_name="Johnson",
                email_address="robert.johnson@example.com",
                phone_number="555-9012",
                address_line1="789 Pine Blvd",
                city="Anotherville",
                state="TX",
                postal_code="67890",
                country="US",
                latitude=29.7604,
                longitude=-95.3698
            )
        ]
        
        with self.db.get_session() as session:
            session.add_all(customers)
            session.commit()
    
    def test_search_customers_by_name(self):
        """Test searching customers by name."""
        # Prepare parameters as expected by run
        kwargs_str = json.dumps({"query": "John Smith", "category": "name"})
        
        # Call the run method as would be done in production
        result = self.tool.run("search_customers", kwargs=kwargs_str)
        
        # Check result structure
        self.assertIn("search_type", result)
        self.assertIn("customers", result)
        self.assertEqual(result["search_type"], "Full Name Search")
        
        # Check found customers
        self.assertEqual(len(result["customers"]), 1)
        self.assertEqual(result["customers"][0]["id"], "cust1")
        self.assertEqual(result["customers"][0]["given_name"], "John")
        self.assertEqual(result["customers"][0]["family_name"], "Smith")
    
    def test_search_customers_by_email(self):
        """Test searching customers by email."""
        kwargs_str = json.dumps({"query": "jane.doe", "category": "email"})
        result = self.tool.run("search_customers", kwargs=kwargs_str)
        
        self.assertEqual(result["search_type"], "Email Address Search")
        self.assertEqual(len(result["customers"]), 1)
        self.assertEqual(result["customers"][0]["id"], "cust2")
    
    def test_search_customers_by_partial_address(self):
        """Test searching customers by partial address."""
        kwargs_str = json.dumps({"query": "pine", "category": "address"})
        result = self.tool.run("search_customers", kwargs=kwargs_str)
        
        self.assertEqual(result["search_type"], "Address Search")
        self.assertEqual(len(result["customers"]), 1)
        self.assertEqual(result["customers"][0]["id"], "cust3")
    
    def test_get_customer_by_id(self):
        """Test getting a customer by ID."""
        kwargs_str = json.dumps({"customer_id": "cust2"})
        result = self.tool.run("get_customer", kwargs=kwargs_str)
        
        self.assertIn("customer", result)
        self.assertEqual(result["customer"]["id"], "cust2")
        self.assertEqual(result["customer"]["given_name"], "Jane")
    
    def test_get_customer_not_found(self):
        """Test error when customer ID not found."""
        kwargs_str = json.dumps({"customer_id": "nonexistent"})
        
        with self.assertRaises(ToolError) as context:
            self.tool.run("get_customer", kwargs=kwargs_str)
        
        self.assertIn("not found", str(context.exception))
    
    def test_find_closest_customers(self):
        """Test finding customers closest to coordinates."""
        # Coordinates near San Francisco (close to John Smith)
        kwargs_str = json.dumps({
            "lat": 37.77,
            "lng": -122.42,
            "limit": 2
        })
        
        result = self.tool.run("find_closest_customers", kwargs=kwargs_str)
        
        self.assertIn("customers", result)
        self.assertEqual(len(result["customers"]), 2)
        
        # First result should be John Smith (closest)
        self.assertEqual(result["customers"][0]["id"], "cust1")
        self.assertIn("distance_meters", result["customers"][0])
        
        # Results should be sorted by distance
        self.assertLess(
            result["customers"][0]["distance_meters"],
            result["customers"][1]["distance_meters"]
        )
    
    @patch('tools.square_tool.SquareTool')
    def test_rebuild_directory(self, mock_square_tool_class):
        """Test rebuilding the customer directory."""
        # Mock the SquareTool's run method
        mock_square_tool = MagicMock()
        mock_square_tool.run.return_value = {
            "customers": [
                {
                    "id": "new_cust",
                    "given_name": "Alice",
                    "family_name": "Williams",
                    "email_address": "alice.williams@example.com"
                }
            ]
        }
        mock_square_tool_class.return_value = mock_square_tool
        
        # Empty the database first
        with self.db.get_session() as session:
            session.query(Customer).delete()
            session.commit()
        
        # Test the rebuild operation
        kwargs_str = json.dumps({"source": "square"})
        result = self.tool.run("rebuild_directory", kwargs=kwargs_str)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["new_customers"], 1)
        
        # Verify the new customer was added
        with self.db.get_session() as session:
            customer = session.query(Customer).filter_by(id="new_cust").first()
            self.assertIsNotNone(customer)
            self.assertEqual(customer.given_name, "Alice")
            self.assertEqual(customer.family_name, "Williams")


if __name__ == "__main__":
    unittest.main()