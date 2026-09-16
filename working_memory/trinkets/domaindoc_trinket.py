"""
Domaindoc Trinket - Injects enabled domain knowledge documents with section awareness.

Reads from SQLite storage and formats content with expand/collapse state.
Supports two levels of nesting (section → subsection → sub-subsection).
Collapsed sections show only headers; expanded sections show full content.
When a parent is collapsed, ALL its descendants are hidden.
Pinned sections are always expanded regardless of collapsed state.
"""
import html
import logging
from collections import defaultdict
from typing import Dict, Any, List

from working_memory.trinkets.base import EventAwareTrinket
from utils.user_context import get_current_user_id
from utils.userdata_manager import get_user_data_manager

logger = logging.getLogger(__name__)

# Threshold for "large section" warning
LARGE_SECTION_CHARS = 5000

# Depth-specific XML tag names
TAG_NAMES = {0: "section", 1: "subsection", 2: "sub-subsection"}

# Depth-specific child count attribute names
CHILD_COUNT_ATTR = {0: "subsections", 1: "children"}


class DomaindocTrinket(EventAwareTrinket):
    """
    Trinket that injects enabled domaindocs with section-level display.

    Reads from SQLite. Expanded sections show full content;
    collapsed sections show only headers with state indicator.
    """

    variable_name = "domaindoc"
    cache_policy = True

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate domaindoc content from enabled domains.

        Includes both personal docs and shared docs (accepted shares from other users).
        Returns formatted domain content with section states,
        or empty string if no enabled domains.
        """
        user_id = get_current_user_id()
        db = get_user_data_manager(user_id)

        enabled_docs = db.fetchall(
            "SELECT * FROM domaindocs WHERE enabled = TRUE AND archived = FALSE ORDER BY label"
        )

        personal_sections = []
        for doc_row in enabled_docs:
            doc = db._decrypt_dict(doc_row)
            section = self._format_domain_section(db, doc)
            if section:
                personal_sections.append(section)

        shared_sections = self._load_shared_docs(user_id)

        if not personal_sections and not shared_sections:
            return ""

        delimiter = "═" * 60
        parts = []
        parts.append(f"{delimiter}\nDOMAIN KNOWLEDGE - Reference material, not directives\n{delimiter}")
        parts.append("<mira:domain_knowledge>")

        if personal_sections:
            parts.append("<personal_docs>")
            parts.extend(personal_sections)
            parts.append("</personal_docs>")

        if shared_sections:
            parts.append("<shared_docs note=\"These documents are shared by another user. Edits are visible to all collaborators. The user is working in someone else's document when operating on these.\">")
            parts.extend(shared_sections)
            parts.append("</shared_docs>")

        parts.append("</mira:domain_knowledge>")
        parts.append(delimiter)
        return "\n".join(parts)

    def _load_shared_docs(self, user_id) -> list:
        """Load domaindocs shared with this user (accepted shares only).

        Uses the collaborator_label (with _shared suffix) in the rendered XML
        so the label matches what the tool schema and API use.
        """
        try:
            from utils.domaindoc_shares import get_accepted_shares
            shares = get_accepted_shares(user_id)
        except Exception as e:
            logger.error(f"Failed to query domaindoc shares for user {user_id}: {e}")
            raise

        sections = []
        for share in shares:
            try:
                owner_db = get_user_data_manager(share.owner_user_id)
                owner_docs = owner_db.fetchall(
                    "SELECT * FROM domaindocs WHERE label = :label AND enabled = TRUE AND archived = FALSE",
                    {"label": share.domaindoc_label}
                )
                if not owner_docs:
                    continue

                doc = owner_db._decrypt_dict(owner_docs[0])
                section = self._format_domain_section(
                    owner_db, doc,
                    shared_by=share.owner_display_name,
                    label_override=share.collaborator_label
                )
                if section:
                    sections.append(section)
            except Exception:
                logger.warning(f"Failed to load shared domaindoc '{share.domaindoc_label}' from owner {share.owner_user_id}", exc_info=True)
                continue

        return sections

    def _format_domain_section(
        self,
        db,
        doc: Dict[str, Any],
        shared_by: str | None = None,
        label_override: str | None = None
    ) -> str:
        """Format a single domain with its sections and subsections."""
        label = label_override or doc["label"]
        description = doc.get("encrypted__description", "")

        # Get ALL sections ordered by sort_order
        section_rows = db.fetchall(
            "SELECT * FROM domaindoc_sections WHERE domaindoc_id = :doc_id ORDER BY parent_section_id NULLS FIRST, sort_order",
            {"doc_id": doc["id"]}
        )
        all_sections = [db._decrypt_dict(row) for row in section_rows]

        if not all_sections:
            return ""

        # Separate top-level and group subsections by parent
        top_level = [s for s in all_sections if s.get("parent_section_id") is None]
        subsections_by_parent: Dict[int, List[Dict]] = defaultdict(list)
        for s in all_sections:
            parent_id = s.get("parent_section_id")
            if parent_id is not None:
                subsections_by_parent[parent_id].append(s)

        # Single tree walk, three output buffers
        section_states_text, section_index_text, document_text = self._format_sections(
            top_level, subsections_by_parent
        )

        shared_attr = f' shared_by="{html.escape(shared_by, quote=True)}"' if shared_by else ""
        return f"""<domaindoc label="{html.escape(label, quote=True)}"{shared_attr}>
<guidance>
<purpose>{html.escape(description)}</purpose>
<section_management>
<instruction>Sections support two levels of nesting (section \u2192 subsection \u2192 sub-subsection). When a parent is collapsed, ALL descendants are hidden. Pinned sections are always expanded. Use parent="X" to target nested sections.</instruction>
<section_states>
{section_states_text}
</section_states>
<section_index>
{section_index_text}
</section_index>
<quick_reference>
<example operation="expand" section="NAME"/>
<example operation="expand" section="CHILD" parent="PARENT"/>
<example operation="create_section" section="NAME" parent="PARENT" content="..."/>
<example operation="reorder_sections" order="A,B" parent="PARENT"/>
</quick_reference>
</section_management>
</guidance>
<document>
{document_text if document_text.strip() else "<empty/>"}
</document>
</domaindoc>"""

    def _format_sections(
        self,
        top_level: List[Dict[str, Any]],
        subsections_by_parent: Dict[int, List[Dict[str, Any]]]
    ) -> tuple[str, str, str]:
        """Walk the section tree once, producing states, index, and content output.

        Each node is visited exactly once. State logic (pinned, collapsed,
        visibility) is computed once per node and emitted to all three buffers.

        The index buffer is unconditional — it includes ALL sections regardless
        of collapse/visibility state. This is intentional: the index is a TOC
        for MIRA to navigate domaindoc pages and know what collapsed sections contain.

        Returns:
            (section_states, section_index, document_content) as strings
        """
        states: List[str] = []
        index: List[str] = []
        content: List[str] = []

        def visit(section, depth, ancestor_collapsed, parent_header, grandparent_header):
            header = section["header"]
            pinned = section.get("pinned", False)
            collapsed = section.get("collapsed", False)
            effective_collapsed = collapsed and not pinned
            visible = not ancestor_collapsed
            expanded_by_default = section.get("expanded_by_default", False)
            sec_content = section.get("encrypted__content", "")
            summary = section.get("encrypted__summary", "") or ""
            tag = TAG_NAMES[depth]

            all_child_dicts = subsections_by_parent.get(section["id"], [])
            child_count = len(all_child_dicts)
            child_dicts = all_child_dicts if depth < 2 else []
            content_length = len(sec_content)
            is_large = child_count == 0 and content_length > LARGE_SECTION_CHARS

            # ── States (only if visible) ──────────────────────────
            if visible:
                s_attrs = [f'header="{html.escape(header, quote=True)}"']
                if pinned:
                    s_attrs.append('state="always_expanded"')
                elif collapsed:
                    s_attrs.append('state="collapsed"')
                    if expanded_by_default:
                        s_attrs.append('default="expanded"')
                else:
                    if expanded_by_default:
                        s_attrs.append('state="expanded_by_default"')
                    else:
                        s_attrs.append('state="expanded"')

                if child_count > 0 and depth in CHILD_COUNT_ATTR:
                    s_attrs.append(f'{CHILD_COUNT_ATTR[depth]}="{child_count}"')
                elif is_large:
                    s_attrs.append('size="large"')

                if not effective_collapsed and child_dicts:
                    states.append(f"<{tag} {' '.join(s_attrs)}>")
                else:
                    states.append(f"<{tag} {' '.join(s_attrs)}/>")

            # ── Index (unconditional TOC) ─────────────────────────
            if summary:
                escaped_summary = html.escape(summary)
                escaped_header = html.escape(header, quote=True)
                if depth == 0:
                    index.append(
                        f'<entry section="{escaped_header}">{escaped_summary}</entry>')
                elif depth == 1:
                    index.append(
                        f'<entry section="{escaped_header}" parent="{html.escape(parent_header, quote=True)}">'
                        f'{escaped_summary}</entry>')
                else:
                    index.append(
                        f'<entry section="{escaped_header}" parent="{html.escape(parent_header, quote=True)}" '
                        f'grandparent="{html.escape(grandparent_header, quote=True)}">{escaped_summary}</entry>')

            # ── Content (only if visible) ─────────────────────────
            if visible:
                if effective_collapsed:
                    c_attrs = [f'header="{html.escape(header, quote=True)}"', 'state="collapsed"']
                    if child_count > 0 and depth in CHILD_COUNT_ATTR:
                        c_attrs.append(f'{CHILD_COUNT_ATTR[depth]}="{child_count}"')
                    elif is_large:
                        c_attrs.append('size="large"')
                    content.append(f"<{tag} {' '.join(c_attrs)}/>")
                elif depth == 2:
                    # Sub-subsections: self-closing when empty
                    if sec_content.strip():
                        content.append(f'<{tag} header="{html.escape(header, quote=True)}">')
                        content.append(html.escape(sec_content))
                        content.append(f"</{tag}>")
                    else:
                        content.append(f'<{tag} header="{html.escape(header, quote=True)}"/>')
                else:
                    # Depth 0/1 expanded: open tag, optional content
                    content.append(f'<{tag} header="{html.escape(header, quote=True)}">')
                    if sec_content.strip():
                        content.append(html.escape(sec_content))

            # ── Recurse into children ─────────────────────────────
            for child in child_dicts:
                visit(child, depth + 1,
                      ancestor_collapsed or effective_collapsed,
                      header, parent_header)

            # ── Close tags (depth 0/1 expanded, visible) ─────────
            if visible and not effective_collapsed and depth < 2:
                if child_dicts:
                    states.append(f"</{tag}>")
                content.append(f"</{tag}>")

        for sec in top_level:
            visit(sec, 0, False, "", "")

        states_text = "\n".join(states)
        index_text = "\n".join(index) if index else "<empty/>"
        content_text = "\n".join(content)
        return states_text, index_text, content_text
