"""
Tests for the Square via HTTP tool.

These tests validate that the Square API integration works correctly
by mocking HTTP responses.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from tools.squareviahttp_tool import SquareViaHttpTool
from errors import ToolError

class TestSquareViaHttpTool(unittest.TestCase):
    """Test case for the SquareViaHttpTool class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create an instance of the tool
        self.tool = SquareViaHttpTool()
        
        # Mock the HTTPTool run method
        self.http_tool_patch = patch('tools.http_tool.HTTPTool.run')
        self.mock_http_run = self.http_tool_patch.start()
        
        # Store the original environ to restore after tests
        self.original_environ = os.environ.copy()
        
        # Set a mock API key for testing
        os.environ['SQUARE_API_KEY'] = 'test_api_key'
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patch
        self.http_tool_patch.stop()
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_environ)
    
    def test_list_customers(self):
        """Test the list_customers operation."""
        # Mock HTTP response
        self.mock_http_run.return_value = {
            'success': True,
            'status_code': 200,
            'data': {
                'customers': [
                    {
                        'id': 'TEST_ID_1',
                        'given_name': 'John',
                        'family_name': 'Doe'
                    }
                ]
            }
        }
        
        # Execute the operation
        result = self.tool.run(operation='list_customers')
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['status_code'], 200)
        self.assertIn('customers', result['data'])
        self.assertEqual(len(result['data']['customers']), 1)
        self.assertEqual(result['data']['customers'][0]['given_name'], 'John')
        
        # Verify the HTTP call was made with correct parameters
        self.mock_http_run.assert_called_once()
        call_args = self.mock_http_run.call_args[1]
        self.assertEqual(call_args['method'], 'GET')
        self.assertTrue(call_args['url'].endswith('/v2/customers'))
        self.assertEqual(call_args['headers']['Authorization'], 'Bearer test_api_key')
    
    def test_create_customer(self):
        """Test the create_customer operation."""
        # Mock HTTP response
        self.mock_http_run.return_value = {
            'success': True,
            'status_code': 200,
            'data': {
                'customer': {
                    'id': 'NEW_CUSTOMER_ID',
                    'given_name': 'Jane',
                    'family_name': 'Smith',
                    'email_address': 'jane.smith@example.com'
                }
            }
        }
        
        # Test data
        customer_data = {
            'given_name': 'Jane',
            'family_name': 'Smith',
            'company_name': 'ACME Inc',
            'email_address': 'jane.smith@example.com',
            'phone_number': '+15551234567',
            'address': {
                'address_line_1': '123 Main St',
                'locality': 'Anytown',
                'administrative_district_level_1': 'CA',
                'postal_code': '94103',
                'country': 'US'
            }
        }
        
        # Execute the operation
        result = self.tool.run(operation='create_customer', **customer_data)
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['customer']['email_address'], 'jane.smith@example.com')
        
        # Verify the HTTP call
        self.mock_http_run.assert_called_once()
        call_args = self.mock_http_run.call_args[1]
        self.assertEqual(call_args['method'], 'POST')
        self.assertTrue(call_args['url'].endswith('/v2/customers'))
        self.assertEqual(call_args['json']['customer']['given_name'], 'Jane')
    
    def test_mark_appointment_paid_by_check(self):
        """Test the mark_appointment_paid_by_check operation."""
        # Mock HTTP response
        self.mock_http_run.return_value = {
            'success': True,
            'status_code': 200,
            'data': {
                'payment': {
                    'id': 'PAYMENT_ID',
                    'amount_money': {
                        'amount': 2500,
                        'currency': 'USD'
                    },
                    'status': 'COMPLETED',
                    'source_type': 'EXTERNAL',
                    'order_id': 'ORDER_ID'
                }
            }
        }
        
        # Execute the operation
        result = self.tool.run(
            operation='mark_appointment_paid_by_check',
            order_id='ORDER_ID',
            amount=25.00,
            description='Check payment'
        )
        
        # Verify the result
        self.assertTrue(result['success'])
        self.assertEqual(result['data']['payment']['id'], 'PAYMENT_ID')
        self.assertEqual(result['data']['payment']['amount_money']['amount'], 2500)
        
        # Verify the HTTP call
        self.mock_http_run.assert_called_once()
        call_args = self.mock_http_run.call_args[1]
        self.assertEqual(call_args['method'], 'POST')
        self.assertTrue(call_args['url'].endswith('/v2/payments'))
        self.assertEqual(call_args['json']['source_id'], 'EXTERNAL')
        self.assertEqual(call_args['json']['amount_money']['amount'], 2500)
        self.assertEqual(call_args['json']['order_id'], 'ORDER_ID')
        self.assertEqual(call_args['json']['external_details']['type'], 'OTHER')
    
    def test_error_handling(self):
        """Test error handling in the tool."""
        # Mock HTTP error response
        self.mock_http_run.return_value = {
            'success': False,
            'status_code': 400,
            'data': {
                'errors': [
                    {
                        'category': 'INVALID_REQUEST_ERROR',
                        'code': 'BAD_REQUEST',
                        'detail': 'Invalid request'
                    }
                ]
            }
        }
        
        # Verify that an error is raised
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation='list_customers')
        
        # Verify error details
        self.assertIn('Square API error', str(context.exception))
    
    def test_invalid_operation(self):
        """Test handling of invalid operations."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation='invalid_operation')
        
        self.assertIn('Unsupported Square operation', str(context.exception))
    
    def test_missing_required_param(self):
        """Test validation of required parameters."""
        with self.assertRaises(ToolError) as context:
            self.tool.run(operation='retrieve_customer')  # Missing customer_id
        
        self.assertIn('Missing required parameter', str(context.exception))


if __name__ == '__main__':
    unittest.main()