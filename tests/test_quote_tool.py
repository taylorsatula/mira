import unittest
from unittest.mock import patch, Mock
from tools.quote_tool import QuoteTool
from errors import ToolError


class TestQuoteTool(unittest.TestCase):

    def setUp(self):
        self.quote_tool = QuoteTool()
        
        # Sample HTML response that simulates the quote tool response
        self.sample_html_response = """
        <html>
        <body>
        <p>A NEW QUOTE HAS BEEN GENERATED
        <br><br>
        TOTAL CUSTOMER-FACING PRICE: <br>
        <h1>
            <span class="dollarsine">$</span>255.50
        </h1><br><br>
        <p>ESTIMATED BILLABLE TIME:</p>
        <h2>2 hours<br>30 minutes</h2>
        <br><br>
        </body>
        </html>
        """

    @patch('requests.post')
    def test_quote_tool_success(self, mock_post):
        # Set up the mock to return our sample HTML
        mock_response = Mock()
        mock_response.text = self.sample_html_response
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Test parameters
        params = {
            "destination_address": "123 Main St, Huntsville, AL",
            "windows_easy": 10,
            "windows_standard": 5,
            "windows_time_consuming": 2,
            "windows_transom": 1,
            "windows_storm": 3,
            "screen_covering": 8
        }

        # Call the tool
        result = self.quote_tool(**params)

        # Verify the correct URL and data were used
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        self.assertEqual(
            call_args['url'],
            "https://webapps.rocketcitywindowcleaning.com/quote-v2/v2_variables.php"
        )
        
        # Check that all parameters were passed correctly
        self.assertEqual(call_args['data']['destinationAddress'], "123 Main St, Huntsville, AL")
        self.assertEqual(call_args['data']['windowsEasy'], 10)
        self.assertEqual(call_args['data']['screenCovering'], 8)

        # Check the result
        self.assertEqual(result['price'], 255.50)
        self.assertEqual(result['estimated_time_minutes'], 150)  # 2 hours 30 minutes = 150 minutes
        self.assertEqual(result['hours'], 2)
        self.assertEqual(result['minutes'], 30)

    @patch('requests.post')
    def test_quote_tool_missing_required_parameter(self, mock_post):
        # Test with missing required parameter
        params = {
            "destination_address": "123 Main St, Huntsville, AL",
            # Missing windows_easy and others
        }

        # Verify that the tool raises an error for missing parameters
        with self.assertRaises(ToolError):
            self.quote_tool(**params)

    @patch('requests.post')
    def test_quote_tool_request_error(self, mock_post):
        # Set up the mock to raise an exception
        mock_post.side_effect = Exception("Network error")

        # Test parameters
        params = {
            "destination_address": "123 Main St, Huntsville, AL",
            "windows_easy": 10,
            "windows_standard": 5,
            "windows_time_consuming": 2,
            "windows_transom": 1,
            "windows_storm": 3,
            "screen_covering": 8
        }

        # Verify that the tool handles request errors properly
        with self.assertRaises(ToolError):
            self.quote_tool(**params)


if __name__ == '__main__':
    unittest.main()