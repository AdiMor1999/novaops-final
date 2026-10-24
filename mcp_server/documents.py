"""Read the NovaOps documents: metadata, sections and chunks.

Starting code: Lesson 7 `02-filtering/ingest.py` (`parse_frontmatter`, `chunk_document`).
Ingest (`scripts/ingest.py`) and `get_policy` both read documents through this module, so
a chunk's `[source#section]` label and the policy tool's label are built the same way.

Only `handbook/` and `manager_playbook/` carry YAML front matter. The other collections
start with a title and inline lines such as `Effective date: 2026-06-01`, so metadata
comes from whichever of the two a file has. Front matter is metadata, not content: it
is stripped before chunking and never embedded.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

CORPORA = (
    "contracts",
    "employment",
    "handbook",
    "internal_memos",
    "it_kb",
    "manager_playbook",
    "policies",
)

# Order matters: the first key a file has wins. Policies say "Effective date", memos
# "Date", IT KB articles "Last updated".
INLINE_DATE_KEYS = ("effective date", "date", "last updated")
EMPLOYEE_NAME_KEYS = ("employee name", "employee")

# A section longer than this is split into overlapping windows. Most sections are well
# under it; the handbook and manager playbook have a few long ones.
CHUNK_WORDS = 300
OVERLAP_WORDS = 50

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_INLINE_FIELD = re.compile(r"^([A-Za-z][A-Za-z _-]{0,40}):\s*(.+)$")


@dataclass
class Document:
    corpus: str
    source: str  # file name, e.g. equipment_policy.md
    title: str
    body: str  # front matter stripped
    audience: str | None = None
    last_updated: str | None = None
    fields: dict = field(default_factory=dict)  # inline "Key: value" header lines

    @property
    def doc_id(self) -> str:
        return f"{self.corpus}/{self.source}"

    @property
    def doc_type(self) -> str:
        return {"internal_memos": "memo", "policies": "policy"}.get(self.corpus, self.corpus)

    @property
    def employee_name(self) -> str | None:
        for key in EMPLOYEE_NAME_KEYS:
            if key in self.fields:
                return self.fields[key]
        return None


@dataclass
class Chunk:
    doc: Document
    n: int
    section: str
    text: str
    subject_employee_id: str | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc.doc_id}#{self.n}"

    @property
    def citation(self) -> str:
        return citation_label(self.doc.source, self.section)

    def record(self, embedding_model: str) -> dict:
        """The OpenSearch `_source` for this chunk, without the vector."""
        rec = {
            "text": self.text,
            "source": self.doc.source,
            "corpus": self.doc.corpus,
            "doc_id": self.doc.doc_id,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "title": self.doc.title,
            "doc_type": self.doc.doc_type,
            "citation": self.citation,
            "embedding_model": embedding_model,
        }
        # Optional fields are left out rather than stored as null, so "no audience"
        # means the field is absent, which is what the must_not filter expects.
        if self.doc.audience:
            rec["audience"] = self.doc.audience
        if self.doc.last_updated:
            rec["last_updated"] = self.doc.last_updated
        if self.subject_employee_id:
            rec["subject_employee_id"] = self.subject_employee_id
        return rec


def citation_label(source: str, section: str) -> str:
    return f"[{source}#{section}]"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a '---'-fenced header from the body. Flat 'key: value' lines only."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    return meta, parts[2].lstrip("\n")


def inline_fields(body: str) -> dict:
    """'Key: value' lines in the header, before the first '## ' section."""
    found = {}
    for line in body.splitlines():
        if line.startswith("## "):
            break
        match = _INLINE_FIELD.match(line.strip())
        if match:
            found.setdefault(match.group(1).strip().lower(), match.group(2).strip())
    return found


def _title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _last_updated(meta: dict, fields: dict) -> str | None:
    candidates = [meta.get("last_updated")] + [fields.get(key) for key in INLINE_DATE_KEYS]
    for value in candidates:
        if value:
            match = _DATE.search(value)
            if match:
                return match.group(0)
    return None


def parse_document(path: Path, corpus: str) -> Document:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    fields = inline_fields(body)
    return Document(
        corpus=corpus,  # the folder, for every file; front matter agrees where it has one
        source=path.name,
        title=_title(body, path.stem),
        body=body,
        audience=meta.get("audience") or None,
        last_updated=_last_updated(meta, fields),
        fields=fields,
    )


def load_corpus(documents_dir: Path, corpus: str) -> list[Document]:
    return [parse_document(p, corpus) for p in sorted((documents_dir / corpus).glob("*.md"))]


def split_sections(doc: Document) -> list[tuple[str, str]]:
    """Split on '## ' headings. The text before the first heading (title and header
    lines) becomes a section named after the document title, so every chunk has a
    section for its label. Files with no '## ' at all are one section."""
    sections: list[tuple[str, list[str]]] = [(doc.title, [])]
    for line in doc.body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    result = []
    for name, lines in sections:
        text = "\n".join(lines).strip()
        if text and text != f"# {doc.title}":
            result.append((name, text))
    return result


def window(text: str) -> list[str]:
    """Lesson 7's sliding word window, used only inside one long section."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    pieces, start = [], 0
    while start < len(words):
        pieces.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_WORDS - OVERLAP_WORDS
    return pieces


def chunk_document(doc: Document) -> list[Chunk]:
    """Chunks of one file only, never across files. The embedded text starts with the
    title and section, so a short chunk still says what it is about."""
    chunks = []
    for section, text in split_sections(doc):
        for piece in window(text):
            body = f"{doc.title} / {section}\n\n{piece}"
            chunks.append(Chunk(doc=doc, n=len(chunks), section=section, text=body))
    return chunks


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def match_employee(doc: Document, employees: dict[str, str]) -> str | None:
    """Employee id for a person-specific employment file, or None.

    `employees` maps full name to employee id. The `Employee name:` / `Employee:` line
    wins; otherwise the file name must start with exactly one employee's full name.
    No partial or first-name matching: the seed has two Daniels, two Mayas and two
    Saras, and a wrong subject is worse than none.
    """
    by_norm = {_normalize(name): eid for name, eid in employees.items()}
    if doc.employee_name:
        return by_norm.get(_normalize(doc.employee_name))
    stem = _normalize(Path(doc.source).stem)
    hits = {eid for norm, eid in by_norm.items() if stem.startswith(norm)}
    return hits.pop() if len(hits) == 1 else None


def build_chunks(documents_dir: Path, employees: dict[str, str]) -> list[Chunk]:
    chunks = []
    for corpus in CORPORA:
        for doc in load_corpus(documents_dir, corpus):
            subject = None
            if corpus == "employment":
                subject = match_employee(doc, employees)
                if subject is None:
                    log.warning("employment/%s matches no employee; subject left empty", doc.source)
            for chunk in chunk_document(doc):
                chunk.subject_employee_id = subject
                chunks.append(chunk)
    return chunks
