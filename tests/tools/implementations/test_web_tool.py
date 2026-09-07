"""
Tests for WebTool.

Following MIRA's real testing philosophy:
- Test contracts, not implementation
- Verify exact return structures and error messages
- Cover input validation and security boundaries
"""
import pytest
from pydantic import ValidationError

from tools.implementations.web_tool import (
    WebTool,
    WebToolConfig,
    SearchInput,
    FetchInput,
    HttpInput,
)


class TestWebToolContract:
    """Tests that enforce WebTool's contract guarantees."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_tool_name_and_schema(self, web_tool):
        """Verify tool name matches schema name."""
        assert web_tool.name == "web_tool"
        assert web_tool.anthropic_schema["name"] == "web_tool"

    def test_schema_has_all_operations(self, web_tool):
        """Verify schema includes all operations."""
        ops = web_tool.anthropic_schema["input_schema"]["properties"]["operation"]["enum"]
        assert "search" in ops
        assert "fetch" in ops
        assert "http" in ops

    def test_unknown_operation_raises_valueerror(self, web_tool, authenticated_user):
        """CONTRACT: Unknown operation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operation:"):
            web_tool.run(operation="invalid_operation")

    def test_missing_operation_raises_valueerror(self, web_tool, authenticated_user):
        """CONTRACT: Missing operation raises ValueError."""
        with pytest.raises(ValueError, match="Required parameter 'operation'"):
            web_tool.run(url="https://example.com")


class TestSearchInputValidation:
    """Tests for search operation input validation."""

    def test_search_input_requires_query(self):
        """CONTRACT: query is required."""
        with pytest.raises(ValidationError):
            SearchInput()

    def test_search_input_rejects_empty_query(self):
        """CONTRACT: Empty query rejected."""
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            SearchInput(query="")

    def test_search_input_validates_max_results_range(self):
        """CONTRACT: max_results must be 1-20."""
        with pytest.raises(ValidationError):
            SearchInput(query="test", max_results=0)
        with pytest.raises(ValidationError):
            SearchInput(query="test", max_results=100)

    def test_search_input_accepts_valid_params(self):
        """CONTRACT: Valid search params accepted."""
        input = SearchInput(query="python tutorial", max_results=10)
        assert input.query == "python tutorial"
        assert input.max_results == 10

    def test_search_input_domain_lists(self):
        """CONTRACT: Domain lists are optional."""
        input = SearchInput(
            query="test",
            allowed_domains=["example.com"],
            blocked_domains=["spam.com"]
        )
        assert input.allowed_domains == ["example.com"]
        assert input.blocked_domains == ["spam.com"]


class TestFetchInputValidation:
    """Tests for fetch operation input validation."""

    def test_fetch_input_requires_url(self):
        """CONTRACT: url is required."""
        with pytest.raises(ValidationError):
            FetchInput()

    def test_fetch_input_validates_url_scheme(self):
        """CONTRACT: Only http/https allowed."""
        with pytest.raises(ValidationError, match="http or https"):
            FetchInput(url="ftp://example.com")
        with pytest.raises(ValidationError, match="http or https"):
            FetchInput(url="file:///etc/passwd")

    def test_fetch_input_accepts_valid_url(self):
        """CONTRACT: Valid HTTP URLs accepted."""
        input = FetchInput(url="https://example.com/page")
        assert input.url == "https://example.com/page"

    def test_fetch_input_has_defaults(self):
        """CONTRACT: Default values for optional fields."""
        input = FetchInput(url="https://example.com")
        assert input.format == "text"
        assert input.include_metadata is False
        assert input.timeout is None

    def test_fetch_input_format_options(self):
        """CONTRACT: Only text/markdown/html formats allowed."""
        FetchInput(url="https://example.com", format="text")
        FetchInput(url="https://example.com", format="markdown")
        FetchInput(url="https://example.com", format="html")

        with pytest.raises(ValidationError):
            FetchInput(url="https://example.com", format="json")


class TestHttpInputValidation:
    """Tests for http operation input validation."""

    def test_http_input_requires_method_and_url(self):
        """CONTRACT: method and url are required."""
        with pytest.raises(ValidationError):
            HttpInput(url="https://api.example.com")
        with pytest.raises(ValidationError):
            HttpInput(method="GET")

    def test_http_input_validates_method(self):
        """CONTRACT: Only GET/POST/PUT/DELETE allowed."""
        HttpInput(method="GET", url="https://api.example.com")
        HttpInput(method="POST", url="https://api.example.com")
        HttpInput(method="PUT", url="https://api.example.com")
        HttpInput(method="DELETE", url="https://api.example.com")

        with pytest.raises(ValidationError):
            HttpInput(method="PATCH", url="https://api.example.com")
        with pytest.raises(ValidationError):
            HttpInput(method="OPTIONS", url="https://api.example.com")

    def test_http_input_validates_url_scheme(self):
        """CONTRACT: Only http/https allowed."""
        with pytest.raises(ValidationError, match="http or https"):
            HttpInput(method="GET", url="ftp://example.com")

    def test_http_input_response_format_options(self):
        """CONTRACT: json/text/full response formats."""
        HttpInput(method="GET", url="https://api.example.com", response_format="json")
        HttpInput(method="GET", url="https://api.example.com", response_format="text")
        HttpInput(method="GET", url="https://api.example.com", response_format="full")

        with pytest.raises(ValidationError):
            HttpInput(method="GET", url="https://api.example.com", response_format="xml")


class TestSSRFProtection:
    """Tests for SSRF (Server-Side Request Forgery) protection."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_blocks_localhost(self, web_tool, authenticated_user):
        """CONTRACT: localhost blocked for security."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://localhost/api")
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://localhost:8080/api")

    def test_blocks_127_addresses(self, web_tool, authenticated_user):
        """CONTRACT: 127.x.x.x addresses blocked."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://127.0.0.1/api")
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://127.0.0.2:3000/api")

    def test_blocks_private_10_network(self, web_tool, authenticated_user):
        """CONTRACT: 10.x.x.x (Class A private) blocked."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://10.0.0.1/internal")
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://10.255.255.255/")

    def test_blocks_private_172_network(self, web_tool, authenticated_user):
        """CONTRACT: 172.16-31.x.x (Class B private) blocked."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://172.16.0.1/")
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://172.31.255.255/")

    def test_blocks_private_192_network(self, web_tool, authenticated_user):
        """CONTRACT: 192.168.x.x (Class C private) blocked."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://192.168.1.1/")
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://192.168.0.100:8080/")

    def test_blocks_link_local(self, web_tool, authenticated_user):
        """CONTRACT: 169.254.x.x (link-local) blocked."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://169.254.1.1/")


class TestDomainRestrictions:
    """Tests for domain allowlist/blocklist functionality."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_allowlist_blocks_unlisted_domains(self, web_tool, authenticated_user):
        """CONTRACT: If allowlist provided, unlisted domains rejected."""
        with pytest.raises(ValueError, match="not in allowed list"):
            web_tool.run(
                operation="http",
                method="GET",
                url="https://blocked.example.com/api",
                allowed_domains=["api.trusted.com"]
            )

    def test_blocklist_blocks_listed_domains(self, web_tool, authenticated_user):
        """CONTRACT: Blocklisted domains rejected."""
        with pytest.raises(ValueError, match="blocked"):
            web_tool.run(
                operation="http",
                method="GET",
                url="https://spam.example.com/api",
                blocked_domains=["spam.example.com"]
            )

    def test_subdomain_matching(self, web_tool):
        """CONTRACT: Domain matching includes subdomains."""
        # Test internal method directly
        assert web_tool._domain_matches("api.example.com", "example.com") is True
        assert web_tool._domain_matches("deep.sub.example.com", "example.com") is True
        assert web_tool._domain_matches("example.com", "example.com") is True
        assert web_tool._domain_matches("notexample.com", "example.com") is False


class TestSearchOperation:
    """Tests for search operation behavior."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_search_without_kagi_key_raises_valueerror(self, web_tool, authenticated_user):
        """CONTRACT: Search requires Kagi API key."""
        # If Kagi isn't configured, should raise helpful error
        if web_tool._kagi is None:
            with pytest.raises(ValueError, match="Kagi API key"):
                web_tool.run(operation="search", query="test query")


class TestFetchOperation:
    """Tests for fetch operation behavior."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_fetch_validates_url(self, web_tool, authenticated_user):
        """CONTRACT: Fetch validates URL before request."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="fetch", url="http://127.0.0.1/secret")


class TestHttpOperation:
    """Tests for http operation behavior."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_http_validates_url(self, web_tool, authenticated_user):
        """CONTRACT: HTTP validates URL before request."""
        with pytest.raises(ValueError, match="private network"):
            web_tool.run(operation="http", method="GET", url="http://10.0.0.1/internal")


class TestArchitecturalConstraints:
    """Test architectural requirements and constraints."""

    def test_tool_extends_base_class(self):
        """CONTRACT: Tool extends Tool base class."""
        from tools.implementations.web_tool import WebTool
        from tools.repo import Tool

        assert issubclass(WebTool, Tool)

    def test_configuration_pydantic(self):
        """CONTRACT: Configuration via Pydantic BaseModel."""
        from tools.implementations.web_tool import WebToolConfig
        from pydantic import BaseModel

        assert issubclass(WebToolConfig, BaseModel)

    def test_config_has_required_fields(self):
        """CONTRACT: Config has standard fields."""
        config = WebToolConfig()
        assert hasattr(config, "enabled")
        assert hasattr(config, "default_timeout")
        assert hasattr(config, "max_timeout")

    def test_no_print_statements(self):
        """CONTRACT: No print statements, only logging."""
        import ast

        file_path = "tools/implementations/web_tool.py"
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    raise AssertionError("Found print statement in implementation")


# ============================================================================
# INTEGRATION TESTS - Real HTTP Calls
# ============================================================================


class TestHttpOperationIntegration:
    """Integration tests for HTTP operation with real network calls."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_get_request_json_response(self, web_tool, authenticated_user):
        """INTEGRATION: GET request returns parsed JSON."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/json",
            response_format="json"
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert "data" in result
        # httpbin.org/json returns a slideshow object
        assert isinstance(result["data"], dict)
        assert "slideshow" in result["data"]

    def test_get_request_text_response(self, web_tool, authenticated_user):
        """INTEGRATION: GET request returns raw text."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/robots.txt",
            response_format="text"
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert "data" in result
        assert isinstance(result["data"], str)

    def test_get_request_full_response(self, web_tool, authenticated_user):
        """INTEGRATION: GET with full format includes headers."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/headers",
            response_format="full"
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert "data" in result
        assert "headers" in result
        assert isinstance(result["headers"], dict)
        # httpx normalizes header names to lowercase
        assert "content-type" in result["headers"]

    def test_get_with_query_params(self, web_tool, authenticated_user):
        """INTEGRATION: GET request with query parameters."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/get",
            params={"foo": "bar", "count": "42"},
            response_format="json"
        )

        assert result["success"] is True
        assert result["data"]["args"]["foo"] == "bar"
        assert result["data"]["args"]["count"] == "42"

    def test_get_with_custom_headers(self, web_tool, authenticated_user):
        """INTEGRATION: GET request with custom headers."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/headers",
            headers={"X-Custom-Header": "test-value"},
            response_format="json"
        )

        assert result["success"] is True
        # httpbin echoes headers back
        assert result["data"]["headers"]["X-Custom-Header"] == "test-value"

    def test_post_request_json_body(self, web_tool, authenticated_user):
        """INTEGRATION: POST request with JSON body."""
        test_data = {"name": "test", "value": 123}
        result = web_tool.run(
            operation="http",
            method="POST",
            url="https://httpbin.org/post",
            json_body=test_data,
            response_format="json"
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        # httpbin echoes JSON back in the "json" field
        assert result["data"]["json"] == test_data

    def test_post_request_form_data(self, web_tool, authenticated_user):
        """INTEGRATION: POST request with form data."""
        form_data = {"username": "testuser", "password": "secret"}
        result = web_tool.run(
            operation="http",
            method="POST",
            url="https://httpbin.org/post",
            data=form_data,
            response_format="json"
        )

        assert result["success"] is True
        # httpbin echoes form data back
        assert result["data"]["form"]["username"] == "testuser"
        assert result["data"]["form"]["password"] == "secret"

    def test_put_request(self, web_tool, authenticated_user):
        """INTEGRATION: PUT request works correctly."""
        result = web_tool.run(
            operation="http",
            method="PUT",
            url="https://httpbin.org/put",
            json_body={"updated": True},
            response_format="json"
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["data"]["json"]["updated"] is True

    def test_delete_request(self, web_tool, authenticated_user):
        """INTEGRATION: DELETE request works correctly."""
        result = web_tool.run(
            operation="http",
            method="DELETE",
            url="https://httpbin.org/delete",
            response_format="json"
        )

        assert result["success"] is True
        assert result["status_code"] == 200

    def test_http_error_status_codes(self, web_tool, authenticated_user):
        """INTEGRATION: HTTP error status codes handled properly."""
        # 404 Not Found
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/status/404",
            response_format="text"
        )

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_redirect_followed(self, web_tool, authenticated_user):
        """INTEGRATION: Redirects are followed automatically."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/redirect/1",
            response_format="json"
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        # Final URL should be /get after redirect
        assert "/get" in result["url"]

    def test_timeout_handling(self, web_tool, authenticated_user):
        """INTEGRATION: Timeout parameter is respected."""
        # Request a 5-second delay with 2-second timeout should fail
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/delay/5",
            timeout=2,
            response_format="json"
        )

        assert result["success"] is False
        assert result["error"] == "timeout"


class TestFetchOperationIntegration:
    """Integration tests for fetch operation with real network calls."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_fetch_simple_page(self, web_tool, authenticated_user):
        """INTEGRATION: Fetch returns content from real webpage."""
        result = web_tool.run(
            operation="fetch",
            url="https://example.com",
            prompt="What is the title and main heading of this page?"
        )

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert "content" in result
        # example.com has minimal content - should mention "Example Domain"
        assert len(result["content"]) > 0

    def test_fetch_with_metadata(self, web_tool, authenticated_user):
        """INTEGRATION: Fetch with metadata returns title and size."""
        result = web_tool.run(
            operation="fetch",
            url="https://example.com",
            prompt="Extract the page content",
            include_metadata=True
        )

        assert result["success"] is True
        assert "title" in result
        assert "metadata" in result
        assert "size" in result["metadata"]
        assert result["metadata"]["size"] > 0

    def test_fetch_httpbin_html(self, web_tool, authenticated_user):
        """INTEGRATION: Fetch HTML content from httpbin."""
        result = web_tool.run(
            operation="fetch",
            url="https://httpbin.org/html",
            prompt="Extract the main text content from this page"
        )

        assert result["success"] is True
        assert "content" in result
        # httpbin/html returns a Herman Melville passage
        assert len(result["content"]) > 0

    def test_fetch_handles_404(self, web_tool, authenticated_user):
        """INTEGRATION: Fetch handles 404 errors gracefully."""
        result = web_tool.run(
            operation="fetch",
            url="https://httpbin.org/status/404",
            prompt="Extract content"
        )

        assert result["success"] is False
        assert "error" in result


class TestSearchOperationIntegration:
    """Integration tests for search operation (requires Kagi API key)."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_search_with_kagi(self, web_tool, authenticated_user):
        """INTEGRATION: Search returns results when Kagi is configured."""
        # Skip if Kagi is not configured
        if web_tool._kagi is None:
            pytest.skip("Kagi API key not configured")

        result = web_tool.run(
            operation="search",
            query="python programming language",
            max_results=3
        )

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) <= 3

        # Each result should have title, url, snippet
        for r in result["results"]:
            assert "title" in r
            assert "url" in r
            assert "snippet" in r

    def test_search_with_domain_filter(self, web_tool, authenticated_user):
        """INTEGRATION: Search respects domain filters."""
        if web_tool._kagi is None:
            pytest.skip("Kagi API key not configured")

        result = web_tool.run(
            operation="search",
            query="python",
            max_results=5,
            allowed_domains=["python.org"]
        )

        assert result["success"] is True
        # All results should be from python.org
        for r in result["results"]:
            assert "python.org" in r["url"].lower()


class TestRealWorldEndpoints:
    """Tests against various real-world endpoints to verify robustness."""

    @pytest.fixture
    def web_tool(self):
        """Create WebTool instance."""
        return WebTool()

    def test_github_api(self, web_tool, authenticated_user):
        """INTEGRATION: GitHub API request works."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://api.github.com/zen",
            response_format="text"
        )

        assert result["success"] is True
        # GitHub Zen returns a random philosophical phrase
        assert len(result["data"]) > 0

    def test_jsonplaceholder_api(self, web_tool, authenticated_user):
        """INTEGRATION: JSONPlaceholder test API works."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://jsonplaceholder.typicode.com/todos/1",
            response_format="json"
        )

        assert result["success"] is True
        assert result["data"]["id"] == 1
        assert "title" in result["data"]
        assert "completed" in result["data"]

    def test_jsonplaceholder_post(self, web_tool, authenticated_user):
        """INTEGRATION: POST to JSONPlaceholder works."""
        result = web_tool.run(
            operation="http",
            method="POST",
            url="https://jsonplaceholder.typicode.com/posts",
            json_body={
                "title": "Test Post",
                "body": "This is a test",
                "userId": 1
            },
            response_format="json"
        )

        assert result["success"] is True
        assert result["status_code"] == 201  # Created
        assert result["data"]["title"] == "Test Post"

    def test_gzip_response(self, web_tool, authenticated_user):
        """INTEGRATION: Gzip-compressed responses handled correctly."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/gzip",
            response_format="json"
        )

        assert result["success"] is True
        assert result["data"]["gzipped"] is True

    def test_unicode_response(self, web_tool, authenticated_user):
        """INTEGRATION: Unicode content handled correctly."""
        result = web_tool.run(
            operation="http",
            method="GET",
            url="https://httpbin.org/encoding/utf8",
            response_format="text"
        )

        assert result["success"] is True
        # Should contain various Unicode characters
        assert "UTF-8" in result["data"] or "encoded" in result["data"].lower()
