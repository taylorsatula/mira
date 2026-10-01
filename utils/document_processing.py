"""
Document processing for multi-turn context.

Processes PDF, DOCX, XLSX, CSV, JSON, and TXT into provider-neutral text
for any fixed model route.

Optional dependencies (python-docx, openpyxl) provide richer extraction
but are not required - falls back to stdlib XML parsing.
"""
import logging
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Optional
from xml.etree import ElementTree

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Try importing optional libraries
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    logger.debug("python-docx not installed - using stdlib fallback for DOCX")

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    logger.debug("openpyxl not installed - using stdlib fallback for XLSX")


SUPPORTED_DOCUMENT_FORMATS = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
    "text/plain",  # TXT
    "text/csv",  # CSV
    "application/json",  # JSON
}

MAX_DOCUMENT_SIZE_MB = 32
MAX_ZIP_ENTRIES = 1000
MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 100
MAX_EXTRACTED_TEXT_CHARS = 1_000_000


@dataclass(frozen=True)
class ProcessedDocument:
    """
    Result of document processing.

    content_type values:
    - "text": Extracted text from every supported format
    """

    content_type: Literal["text"]
    media_type: str     # Original MIME type
    data: str           # extracted_text
    original_filename: Optional[str] = None  # Filename for tracking


def process_document(
    doc_bytes: bytes,
    media_type: str,
    filename: str = "document",
) -> ProcessedDocument:
    """Process a document into provider-neutral text for any fixed model route.

    Type-based routing:
    - CSV/XLSX/JSON → Extract text locally
    - PDF/DOCX → Extract text locally

    Args:
        doc_bytes: Raw document bytes
        media_type: MIME type of the document
        filename: Original filename retained for caller-facing metadata

    Returns:
        ProcessedDocument with appropriate content for LLM

    Raises:
        ValueError: If document type unsupported or processing fails
    """
    # Structured data is extracted locally. Provider-specific remote file IDs
    # cannot cross MIRA's provider-neutral request boundary.
    if media_type in ("text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/json"):
        if media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            validate_office_zip(doc_bytes)
        if media_type == "text/csv":
            text = extract_text_file(doc_bytes)
        elif media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            text = extract_xlsx_text(doc_bytes)
        else:
            text = extract_text_file(doc_bytes)
        return ProcessedDocument(
            content_type="text",
            media_type=media_type,
            data=text,
            original_filename=filename,
        )

    # PDF: extract text locally for the provider-neutral request.
    elif media_type == "application/pdf":
        return ProcessedDocument(
            content_type="text",
            media_type=media_type,
            data=extract_pdf_text(doc_bytes),
            original_filename=filename,
        )

    # DOCX: Extract text locally.
    elif media_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_docx_text(doc_bytes)
        return ProcessedDocument(
            content_type="text",
            media_type=media_type,
            data=text
        )

    # Plain text: Extract as UTF-8
    elif media_type == "text/plain":
        text = extract_text_file(doc_bytes)
        return ProcessedDocument(
            content_type="text",
            media_type=media_type,
            data=text
        )

    else:
        raise ValueError(f"Unsupported document type: {media_type}")


def extract_text_file(doc_bytes: bytes) -> str:
    """
    Extract text from TXT or CSV file.

    Rejects non-UTF-8 bytes rather than decoding them to mojibake, and
    enforces the same extracted-text cap as the PDF/DOCX/XLSX extractors.
    """
    try:
        text = doc_bytes.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError("Document is not valid UTF-8 text") from error
    return _cap_extracted_text(text)


def extract_pdf_text(doc_bytes: bytes) -> str:
    """Extract bounded text from a PDF or reject image-only documents."""
    try:
        reader = PdfReader(BytesIO(doc_bytes))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except Exception as error:
        raise ValueError(f"Failed to extract PDF text: {error}") from error
    text = text.strip()
    if not text:
        raise ValueError("PDF contains no extractable text")
    return _cap_extracted_text(text)


def extract_docx_text(doc_bytes: bytes) -> str:
    """
    Extract text from DOCX document.

    Uses python-docx if available, otherwise falls back to stdlib XML parsing.
    """
    validate_office_zip(doc_bytes)
    if HAS_DOCX:
        text = _extract_docx_with_library(doc_bytes)
    else:
        text = _extract_docx_stdlib(doc_bytes)
    return _cap_extracted_text(text)


def extract_xlsx_text(doc_bytes: bytes) -> str:
    """
    Extract text from XLSX spreadsheet.

    Uses openpyxl if available, otherwise falls back to stdlib XML parsing.
    """
    validate_office_zip(doc_bytes)
    if HAS_OPENPYXL:
        text = _extract_xlsx_with_library(doc_bytes)
    else:
        text = _extract_xlsx_stdlib(doc_bytes)
    return _cap_extracted_text(text)


def validate_office_zip(doc_bytes: bytes) -> None:
    """Reject DOCX/XLSX zip containers that expand beyond safe limits."""
    try:
        with zipfile.ZipFile(BytesIO(doc_bytes)) as z:
            infos = z.infolist()
    except zipfile.BadZipFile as e:
        raise ValueError("Invalid Office document ZIP container") from e

    if len(infos) > MAX_ZIP_ENTRIES:
        raise ValueError(f"Document ZIP has too many entries: {len(infos)}")

    total_uncompressed = 0
    for info in infos:
        if info.file_size > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
            raise ValueError(f"Document ZIP entry too large: {info.filename}")
        total_uncompressed += info.file_size
        if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Document ZIP expands beyond allowed size")
        if info.compress_size > 0 and info.file_size / info.compress_size > MAX_ZIP_COMPRESSION_RATIO:
            raise ValueError(f"Document ZIP compression ratio too high: {info.filename}")
        if info.compress_size == 0 and info.file_size > 0:
            raise ValueError(f"Document ZIP entry has zero compressed size: {info.filename}")


def _cap_extracted_text(text: str) -> str:
    """Enforce a hard extracted-text cap after parsing."""
    if len(text) > MAX_EXTRACTED_TEXT_CHARS:
        raise ValueError("Extracted document text exceeds allowed size")
    return text


def _extract_docx_with_library(doc_bytes: bytes) -> str:
    """Extract DOCX text using python-docx library."""
    try:
        doc = docx.Document(BytesIO(doc_bytes))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {e}") from e


def _extract_docx_stdlib(doc_bytes: bytes) -> str:
    """Extract DOCX text using stdlib only (zipfile + xml)."""
    try:
        with zipfile.ZipFile(BytesIO(doc_bytes)) as z:
            xml_content = z.read('word/document.xml')
            tree = ElementTree.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = [t.text for t in tree.findall('.//w:t', ns) if t.text]
            return ' '.join(texts)
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {e}") from e


def _extract_xlsx_with_library(doc_bytes: bytes) -> str:
    """Extract XLSX text using openpyxl library."""
    try:
        wb = openpyxl.load_workbook(BytesIO(doc_bytes), read_only=True, data_only=True)
        lines = []
        for sheet in wb.worksheets:
            lines.append(f"=== Sheet: {sheet.title} ===")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c) if c is not None else '' for c in row]
                if any(cells):
                    lines.append('\t'.join(cells))
        wb.close()
        return '\n'.join(lines)
    except Exception as e:
        raise ValueError(f"Failed to extract XLSX text: {e}") from e


def _xlsx_sheets_in_order(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Resolve (display name, zip part) for each worksheet, in workbook order.

    `xl/workbook.xml` lists `<sheet name= r:id=>` in the order the user sees;
    `xl/_rels/workbook.xml.rels` maps each rId to its part. Falling back to a
    sorted namelist would both lose the display names and order `sheet10`
    before `sheet2`, so the manifest is authoritative when present.
    """
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    prns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    names = z.namelist()

    def _natural(part: str) -> list:
        stem = part.rsplit('/', 1)[-1]
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', stem)]

    worksheet_parts = sorted(
        (n for n in names if n.startswith('xl/worksheets/') and n.endswith('.xml')),
        key=_natural,
    )

    if 'xl/workbook.xml' in names and 'xl/_rels/workbook.xml.rels' in names:
        try:
            rels_tree = ElementTree.fromstring(z.read('xl/_rels/workbook.xml.rels'))
            rid_to_target = {
                rel.get('Id'): rel.get('Target')
                for rel in rels_tree.findall(f'{{{prns}}}Relationship')
            }
            wb_tree = ElementTree.fromstring(z.read('xl/workbook.xml'))
            ordered = []
            for sheet in wb_tree.findall(f'.//{{{ns}}}sheet'):
                target = rid_to_target.get(sheet.get(f'{{{rns}}}id'))
                if not target:
                    continue
                # Targets are relative to xl/ or absolute from the package root.
                part = target.lstrip('/') if target.startswith('/') else f'xl/{target}'
                if part in names:
                    ordered.append((sheet.get('name') or part.rsplit('/', 1)[-1], part))
            if ordered:
                return ordered
        except ElementTree.ParseError:
            logger.warning("XLSX workbook manifest unparseable; using part-name order")

    return [(part.rsplit('/', 1)[-1], part) for part in worksheet_parts]


def _extract_xlsx_stdlib(doc_bytes: bytes) -> str:
    """Extract XLSX text using stdlib only (zipfile + xml)."""
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    try:
        with zipfile.ZipFile(BytesIO(doc_bytes)) as z:
            # Read shared strings (XLSX stores text in a shared strings table)
            shared_strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                ss_tree = ElementTree.fromstring(z.read('xl/sharedStrings.xml'))
                # One entry per <si>, not per <t>: a rich-text <si> holds several
                # <r><t> runs that concatenate into a single cell value. Counting
                # <t> elements instead would shift every later index.
                for si in ss_tree.findall(f'{{{ns}}}si'):
                    shared_strings.append(
                        ''.join(t.text or '' for t in si.findall(f'.//{{{ns}}}t'))
                    )

            lines = []
            for sheet_name, part in _xlsx_sheets_in_order(z):
                sheet_tree = ElementTree.fromstring(z.read(part))
                lines.append(f"=== Sheet: {sheet_name} ===")
                for row in sheet_tree.findall(f'.//{{{ns}}}row'):
                    cells = []
                    for cell in row.findall(f'{{{ns}}}c'):
                        cell_type = cell.get('t')
                        if cell_type == 'inlineStr':
                            # Inline string: text lives in <is><t>; there is no
                            # <v> and no sharedStrings entry. openpyxl writes
                            # every text cell this way.
                            is_elem = cell.find(f'{{{ns}}}is')
                            cells.append(
                                ''.join(t.text or '' for t in is_elem.findall(f'.//{{{ns}}}t'))
                                if is_elem is not None else ''
                            )
                            continue
                        val = cell.find(f'{{{ns}}}v')
                        if val is not None and val.text:
                            if cell_type == 's':  # Shared string reference
                                idx = int(val.text)
                                cells.append(shared_strings[idx] if idx < len(shared_strings) else '')
                            else:
                                cells.append(val.text)
                        else:
                            cells.append('')
                    if any(cells):
                        lines.append('\t'.join(cells))

            return '\n'.join(lines)
    except Exception as e:
        raise ValueError(f"Failed to extract XLSX text: {e}") from e
