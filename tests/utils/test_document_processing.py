"""
Tests for document_processing.py.

Focus: Real contract guarantees for document processing utility.
"""
import base64
import pytest
import zipfile
from io import BytesIO
from xml.etree import ElementTree

from utils.document_processing import (
    process_document,
    extract_docx_text,
    extract_xlsx_text,
    ProcessedDocument,
    SUPPORTED_DOCUMENT_FORMATS,
    MAX_DOCUMENT_SIZE_MB,
    HAS_DOCX,
    HAS_OPENPYXL,
    _extract_docx_stdlib,
    _extract_xlsx_stdlib,
)


def create_minimal_pdf() -> bytes:
    """Create a minimal valid PDF for testing."""
    # Minimal PDF structure
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF"""
    return pdf_content


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def create_minimal_docx(text: str = "Hello World") -> bytes:
    """Create a minimal valid DOCX file for testing."""
    buffer = BytesIO()
    escaped_text = _escape_xml(text)
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Content Types
        content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''
        zf.writestr('[Content_Types].xml', content_types)

        # Relationships
        rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
        zf.writestr('_rels/.rels', rels)

        # Document
        document = f'''<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p>
            <w:r>
                <w:t>{escaped_text}</w:t>
            </w:r>
        </w:p>
    </w:body>
</w:document>'''
        zf.writestr('word/document.xml', document)

    return buffer.getvalue()


def create_minimal_xlsx(data: list = None) -> bytes:
    """Create a minimal valid XLSX file for testing."""
    if data is None:
        data = [["Name", "Value"], ["Test", "123"]]

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Content Types
        content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
    <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>'''
        zf.writestr('[Content_Types].xml', content_types)

        # Relationships
        rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
        zf.writestr('_rels/.rels', rels)

        # Workbook relationships
        workbook_rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>'''
        zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels)

        # Workbook
        workbook = '''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheets>
        <sheet name="Sheet1" sheetId="1" r:id="rId1" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
    </sheets>
</workbook>'''
        zf.writestr('xl/workbook.xml', workbook)

        # Collect all unique strings
        all_strings = []
        for row in data:
            for cell in row:
                if isinstance(cell, str) and cell not in all_strings:
                    all_strings.append(cell)

        # Shared strings (escape XML special characters)
        si_elements = ''.join(f'<si><t>{_escape_xml(s)}</t></si>' for s in all_strings)
        shared_strings = f'''<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(all_strings)}" uniqueCount="{len(all_strings)}">
{si_elements}
</sst>'''
        zf.writestr('xl/sharedStrings.xml', shared_strings)

        # Sheet data
        rows_xml = []
        for row_idx, row in enumerate(data, 1):
            cells_xml = []
            for col_idx, cell in enumerate(row):
                col_letter = chr(ord('A') + col_idx)
                cell_ref = f"{col_letter}{row_idx}"
                if isinstance(cell, str):
                    # String - reference shared strings
                    str_idx = all_strings.index(cell)
                    cells_xml.append(f'<c r="{cell_ref}" t="s"><v>{str_idx}</v></c>')
                else:
                    # Number
                    cells_xml.append(f'<c r="{cell_ref}"><v>{cell}</v></c>')
            rows_xml.append(f'<row r="{row_idx}">{"".join(cells_xml)}</row>')

        sheet = f'''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
        {"".join(rows_xml)}
    </sheetData>
</worksheet>'''
        zf.writestr('xl/worksheets/sheet1.xml', sheet)

    return buffer.getvalue()


class TestDocxProcessing:
    """Tests for DOCX document processing."""

    def test_docx_returns_text_content_type(self):
        """CONTRACT: DOCX processing returns content_type='text'."""
        docx_bytes = create_minimal_docx("Test content")

        result = process_document(
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assert result.content_type == "text"

    def test_docx_extracts_text_content(self):
        """CONTRACT: DOCX processing extracts text."""
        docx_bytes = create_minimal_docx("Hello World")

        result = process_document(
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assert "Hello" in result.data
        assert "World" in result.data

    def test_docx_stdlib_extracts_text(self):
        """CONTRACT: Stdlib DOCX extraction works without python-docx."""
        docx_bytes = create_minimal_docx("Stdlib Test")

        result = _extract_docx_stdlib(docx_bytes)

        assert "Stdlib" in result
        assert "Test" in result

    def test_docx_handles_xml_special_characters(self):
        """CONTRACT: DOCX with XML special chars (&, <, >) extracts correctly."""
        docx_bytes = create_minimal_docx("Price: $100 & 50% off < today > tomorrow")

        result = process_document(
            docx_bytes,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        assert "&" in result.data
        assert "<" in result.data
        assert ">" in result.data


class TestXlsxProcessing:
    """Tests for XLSX document processing."""

    def test_xlsx_returns_text_content_type(self):
        """CONTRACT: XLSX processing returns content_type='text'."""
        xlsx_bytes = create_minimal_xlsx()

        result = process_document(
            xlsx_bytes,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        assert result.content_type == "text"

    def test_xlsx_extracts_cell_content(self):
        """CONTRACT: XLSX processing extracts cell values."""
        xlsx_bytes = create_minimal_xlsx([["Name", "Value"], ["Alice", "100"]])

        result = process_document(
            xlsx_bytes,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        assert "Name" in result.data
        assert "Value" in result.data
        assert "Alice" in result.data

    def test_xlsx_stdlib_extracts_text(self):
        """CONTRACT: Stdlib XLSX extraction works without openpyxl."""
        xlsx_bytes = create_minimal_xlsx([["Header"], ["Data"]])

        result = _extract_xlsx_stdlib(xlsx_bytes)

        assert "Header" in result
        assert "Data" in result


class TestErrorHandling:
    """Tests for error handling."""

    def test_unsupported_format_raises_valueerror(self):
        """CONTRACT: Unsupported document types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported document type"):
            process_document(b"data", "application/unknown")

    def test_invalid_docx_raises_valueerror(self):
        """CONTRACT: Invalid DOCX data raises ValueError."""
        with pytest.raises(ValueError, match="Failed to extract DOCX"):
            extract_docx_text(b"not a docx file")

    def test_invalid_xlsx_raises_valueerror(self):
        """CONTRACT: Invalid XLSX data raises ValueError."""
        with pytest.raises(ValueError, match="Failed to extract XLSX"):
            extract_xlsx_text(b"not an xlsx file")


class TestSupportedFormats:
    """Tests for supported format configuration."""

    def test_supported_formats_includes_pdf(self):
        """CONTRACT: PDF is a supported format."""
        assert "application/pdf" in SUPPORTED_DOCUMENT_FORMATS

    def test_supported_formats_includes_docx(self):
        """CONTRACT: DOCX is a supported format."""
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in SUPPORTED_DOCUMENT_FORMATS

    def test_supported_formats_includes_xlsx(self):
        """CONTRACT: XLSX is a supported format."""
        assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in SUPPORTED_DOCUMENT_FORMATS

    def test_max_document_size_is_32mb(self):
        """CONTRACT: Max document size is 32MB (Claude's limit)."""
        assert MAX_DOCUMENT_SIZE_MB == 32


class TestRoundTrip:
    """Round-trip tests: create document → extract → verify content."""

    def test_docx_round_trip_preserves_content(self):
        """CONTRACT: Text put into DOCX can be extracted back out."""
        original_text = "The quick brown fox jumps over the lazy dog"
        docx_bytes = create_minimal_docx(original_text)

        # Extract using the main function (uses library or stdlib)
        extracted = extract_docx_text(docx_bytes)

        # Verify all words present
        for word in original_text.split():
            assert word in extracted, f"Missing word: {word}"

    def test_docx_round_trip_stdlib_preserves_content(self):
        """CONTRACT: Stdlib extraction preserves content from created DOCX."""
        original_text = "Testing stdlib extraction pathway"
        docx_bytes = create_minimal_docx(original_text)

        extracted = _extract_docx_stdlib(docx_bytes)

        for word in original_text.split():
            assert word in extracted, f"Missing word: {word}"

    def test_xlsx_round_trip_preserves_content(self):
        """CONTRACT: Data put into XLSX can be extracted back out."""
        original_data = [
            ["Product", "Price", "Quantity"],
            ["Apple", "1.50", "100"],
            ["Banana", "0.75", "200"],
        ]
        xlsx_bytes = create_minimal_xlsx(original_data)

        extracted = extract_xlsx_text(xlsx_bytes)

        # Verify all cell values present
        for row in original_data:
            for cell in row:
                assert cell in extracted, f"Missing cell value: {cell}"

    def test_xlsx_round_trip_stdlib_preserves_content(self):
        """CONTRACT: Stdlib extraction preserves content from created XLSX."""
        original_data = [
            ["Name", "Score"],
            ["Alice", "95"],
            ["Bob", "87"],
        ]
        xlsx_bytes = create_minimal_xlsx(original_data)

        extracted = _extract_xlsx_stdlib(xlsx_bytes)

        for row in original_data:
            for cell in row:
                assert cell in extracted, f"Missing cell value: {cell}"



class TestProcessedDocumentDataclass:
    """Tests for ProcessedDocument dataclass structure."""

    def test_dataclass_has_required_fields(self):
        """CONTRACT: ProcessedDocument has content_type, media_type, data fields."""
        result = ProcessedDocument(content_type="text", media_type="text/csv", data="a,b\n1,2")

        assert hasattr(result, 'content_type')
        assert hasattr(result, 'media_type')
        assert hasattr(result, 'data')

    def test_dataclass_is_frozen(self):
        """CONTRACT: ProcessedDocument is immutable."""
        result = ProcessedDocument(content_type="text", media_type="text/csv", data="a,b\n1,2")

        with pytest.raises(AttributeError):
            result.content_type = "modified"


class TestRoundTrip:
    """Round-trip tests: create document with known content, verify extraction."""

    def test_docx_round_trip_single_paragraph(self):
        """ROUND-TRIP: Single paragraph DOCX content survives extraction."""
        known_content = "The quick brown fox jumps over the lazy dog"
        docx_bytes = create_minimal_docx(known_content)

        result = extract_docx_text(docx_bytes)

        assert known_content in result

    def test_docx_round_trip_special_characters(self):
        """ROUND-TRIP: Special characters survive DOCX extraction."""
        known_content = "Price: $100 & 50% off — \"quoted\" text"
        docx_bytes = create_minimal_docx(known_content)

        result = extract_docx_text(docx_bytes)

        assert "$100" in result
        assert "50%" in result

    def test_docx_round_trip_unicode(self):
        """ROUND-TRIP: Unicode content survives DOCX extraction."""
        known_content = "日本語テスト émojis: 🎉"
        docx_bytes = create_minimal_docx(known_content)

        result = extract_docx_text(docx_bytes)

        assert "日本語" in result

    def test_xlsx_round_trip_string_cells(self):
        """ROUND-TRIP: String cell values survive XLSX extraction."""
        data = [["Name", "City"], ["Alice", "Boston"], ["Bob", "Chicago"]]
        xlsx_bytes = create_minimal_xlsx(data)

        result = extract_xlsx_text(xlsx_bytes)

        assert "Alice" in result
        assert "Boston" in result
        assert "Bob" in result
        assert "Chicago" in result

    def test_xlsx_round_trip_numeric_cells(self):
        """ROUND-TRIP: Numeric cell values survive XLSX extraction."""
        data = [["Item", "Price"], ["Widget", 99.99], ["Gadget", 149.50]]
        xlsx_bytes = create_minimal_xlsx(data)

        result = extract_xlsx_text(xlsx_bytes)

        assert "99.99" in result
        assert "149.5" in result  # Note: trailing zero may be dropped

    def test_xlsx_round_trip_mixed_content(self):
        """ROUND-TRIP: Mixed string/numeric content survives XLSX extraction."""
        data = [["ID", "Name", "Score"], [1, "Test", 85]]
        xlsx_bytes = create_minimal_xlsx(data)

        result = extract_xlsx_text(xlsx_bytes)

        assert "ID" in result
        assert "Name" in result
        assert "Test" in result
        assert "85" in result

    def test_stdlib_docx_matches_content(self):
        """ROUND-TRIP: Stdlib DOCX extraction matches known content."""
        known_content = "Stdlib extraction verification test"
        docx_bytes = create_minimal_docx(known_content)

        result = _extract_docx_stdlib(docx_bytes)

        # All words should be present (may be joined differently)
        for word in ["Stdlib", "extraction", "verification", "test"]:
            assert word in result

    def test_stdlib_xlsx_matches_content(self):
        """ROUND-TRIP: Stdlib XLSX extraction matches known content."""
        data = [["Header1", "Header2"], ["Value1", "Value2"]]
        xlsx_bytes = create_minimal_xlsx(data)

        result = _extract_xlsx_stdlib(xlsx_bytes)

        assert "Header1" in result
        assert "Header2" in result
        assert "Value1" in result
        assert "Value2" in result
