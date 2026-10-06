import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader


DOC_ID = "constitution_nepal_2015"
DOC_TITLE = "Constitution of Nepal"


def extract_text_from_pdf(
    pdf_path: str,
) -> Tuple[str, List[Tuple[int, int]]]:
    """Extract page text and map combined-text offsets to one-based PDF pages."""
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    page_starts: List[Tuple[int, int]] = []
    offset = 0

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"Warning: could not extract PDF page {page_number}: {exc}")
            text = ""

        if not text.strip():
            continue
        page_starts.append((offset, page_number))
        pages.append(text)
        offset += len(text) + 1

    return "\n".join(pages), page_starts


def clean_extracted_text(text: str) -> str:
    """Create search text without changing the separately stored raw text."""
    lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "www.lawcommission.gov.np":
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        lines.append(stripped)

    normalized = " ".join(lines)
    normalized = re.sub(r"(?<=\w)-\s+(?=\w)", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _page_range(
    start: int, end: int, page_starts: List[Tuple[int, int]]
) -> Tuple[Optional[int], Optional[int]]:
    """Return the one-based PDF page range covering a source offset range."""
    if not page_starts or end <= start:
        return None, None
    offsets = [item[0] for item in page_starts]
    first_index = bisect_right(offsets, start) - 1
    last_index = bisect_right(offsets, end - 1) - 1
    if first_index < 0 or last_index < 0:
        return None, None
    return page_starts[first_index][1], page_starts[last_index][1]


def _part_metadata(
    raw_text: str, offset: int
) -> Tuple[Optional[int], Optional[str]]:
    """Find the latest Part heading before an article."""
    part_pattern = re.compile(r"(?mi)^[ \t]*Part[ -](\d+)[ \t]*$")
    part_matches = list(part_pattern.finditer(raw_text, 0, offset))
    if not part_matches:
        return None, None

    part_match = part_matches[-1]
    title_match = re.match(r"[ \t]*\n[ \t]*([^\n]+)", raw_text[part_match.end() :])
    title = title_match.group(1).strip() if title_match else None
    return int(part_match.group(1)), title


def _make_chunk(
    *,
    chunk_id: str,
    parent_chunk_id: Optional[str],
    chunk_type: str,
    raw_text: str,
    source_start: int,
    source_end: int,
    page_starts: List[Tuple[int, int]],
    part_number: Optional[int],
    part_title: Optional[str],
    article_number: Optional[int],
    article_title: Optional[str],
    clause_number: Optional[int] = None,
    subclause_number: Optional[str] = None,
    schedule_number: Optional[int] = None,
    schedule_title: Optional[str] = None,
) -> Dict[str, Any]:
    page_start, page_end = _page_range(source_start, source_end, page_starts)
    raw_provision = raw_text[source_start:source_end].strip()
    return {
        "chunk_id": chunk_id,
        "parent_chunk_id": parent_chunk_id,
        "chunk_type": chunk_type,
        "text": raw_provision,
        "normalized_text": clean_extracted_text(raw_provision),
        "metadata": {
            "doc_id": DOC_ID,
            "doc_title": DOC_TITLE,
            "part_number": part_number,
            "part_title": part_title,
            "chapter_number": None,
            "chapter_title": None,
            "article_number": article_number,
            "article_title": article_title,
            "schedule_number": schedule_number,
            "schedule_title": schedule_title,
            "clause_number": clause_number,
            "subclause_number": subclause_number,
            "page_start": page_start,
            "page_end": page_end,
            "source_start_offset": source_start,
            "source_end_offset": source_end,
        },
    }


def parse_constitution_to_json(
    raw_text: str, page_starts: List[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """Create document-order article parents and reliably detected clause children."""
    chunks: List[Dict[str, Any]] = []
    article_pattern = re.compile(
        r"(?ms)^[ \t]*(\d{1,3})\.[ \t]+"
        r"((?:(?!\n[ \t]*\d{1,3}\.).){1,250}?)"
        r"(?=:\s|\s+\(\d+\)[ \t]+|\s+\([a-z]\)[ \t]+)"
    )
    article_matches = []
    last_article_number = 0
    for match in article_pattern.finditer(raw_text):
        article_number = int(match.group(1))
        if not article_matches:
            if article_number != 1:
                continue
            article_matches.append(match)
            last_article_number = article_number
            continue
        if article_number <= last_article_number:
            break
        article_matches.append(match)
        last_article_number = article_number

    schedule_pattern = re.compile(
        r"(?mi)^[ \t]*(?:\uf0a3[ \t]*)?Schedule[- ]([0-9]+)[ \t]*$"
    )
    schedule_matches = list(schedule_pattern.finditer(raw_text))
    first_schedule_start = (
        schedule_matches[0].start() if schedule_matches else len(raw_text)
    )

    for index, article_match in enumerate(article_matches):
        article_number = int(article_match.group(1))
        article_title = " ".join(article_match.group(2).split())
        article_start = article_match.start()
        article_end = (
            article_matches[index + 1].start()
            if index + 1 < len(article_matches)
            else first_schedule_start
        )
        part_number, part_title = _part_metadata(raw_text, article_start)
        parent_id = f"{DOC_ID}_article_{article_number}"

        chunks.append(
            _make_chunk(
                chunk_id=parent_id,
                parent_chunk_id=None,
                chunk_type="article",
                raw_text=raw_text,
                source_start=article_start,
                source_end=article_end,
                page_starts=page_starts,
                part_number=part_number,
                part_title=part_title,
                article_number=article_number,
                article_title=article_title,
            )
        )

        article_text_start = article_match.end()
        clause_pattern = re.compile(
            r"(?m)(?:(?<=:)|(?<=\n))[ \t]*\((\d+)\)[ \t]+"
        )
        clause_matches = list(
            clause_pattern.finditer(raw_text, article_text_start, article_end)
        )
        reliable_clause_matches = []
        last_clause_number = 0
        for clause_match in clause_matches:
            clause_number = int(clause_match.group(1))
            if clause_number <= last_clause_number:
                break
            reliable_clause_matches.append(clause_match)
            last_clause_number = clause_number

        for clause_index, clause_match in enumerate(reliable_clause_matches):
            clause_number = int(clause_match.group(1))
            clause_start = clause_match.start()
            clause_end = (
                reliable_clause_matches[clause_index + 1].start()
                if clause_index + 1 < len(reliable_clause_matches)
                else article_end
            )
            chunks.append(
                _make_chunk(
                    chunk_id=f"{parent_id}_clause_{clause_number}",
                    parent_chunk_id=parent_id,
                    chunk_type="clause",
                    raw_text=raw_text,
                    source_start=clause_start,
                    source_end=clause_end,
                    page_starts=page_starts,
                    part_number=part_number,
                    part_title=part_title,
                    article_number=article_number,
                    article_title=article_title,
                    clause_number=clause_number,
                )
            )

            clause_text_start = clause_match.end()
            clause_text_end = clause_end
            subclause_pattern = re.compile(
                r"(?m)(?:(?<=:)|(?<=\n))[ \t]*\(([a-z])\)[ \t]+"
            )
            subclause_matches = list(
                subclause_pattern.finditer(
                    raw_text, clause_text_start, clause_text_end
                )
            )
            reliable_subclause_matches = []
            last_subclause = ""
            for subclause_match in subclause_matches:
                subclause_number = subclause_match.group(1).lower()
                if last_subclause and subclause_number <= last_subclause:
                    break
                reliable_subclause_matches.append(subclause_match)
                last_subclause = subclause_number

            for subclause_index, subclause_match in enumerate(
                reliable_subclause_matches
            ):
                subclause_number = subclause_match.group(1).lower()
                subclause_start = subclause_match.start()
                subclause_end = (
                    reliable_subclause_matches[subclause_index + 1].start()
                    if subclause_index + 1 < len(reliable_subclause_matches)
                    else clause_text_end
                )
                chunks.append(
                    _make_chunk(
                        chunk_id=(
                            f"{parent_id}_clause_{clause_number}"
                            f"_subclause_{subclause_number}"
                        ),
                        parent_chunk_id=f"{parent_id}_clause_{clause_number}",
                        chunk_type="subclause",
                        raw_text=raw_text,
                        source_start=subclause_start,
                        source_end=subclause_end,
                        page_starts=page_starts,
                        part_number=part_number,
                        part_title=part_title,
                        article_number=article_number,
                        article_title=article_title,
                        clause_number=clause_number,
                        subclause_number=subclause_number,
                    )
                )

    for index, schedule_match in enumerate(schedule_matches):
        schedule_number = int(schedule_match.group(1))
        schedule_start = schedule_match.start()
        schedule_end = (
            schedule_matches[index + 1].start()
            if index + 1 < len(schedule_matches)
            else len(raw_text)
        )
        schedule_lines = raw_text[schedule_match.end() : schedule_end].splitlines()
        schedule_title = next(
            (" ".join(line.split()) for line in schedule_lines if line.strip()),
            None,
        )
        chunks.append(
            _make_chunk(
                chunk_id=f"{DOC_ID}_schedule_{schedule_number}",
                parent_chunk_id=None,
                chunk_type="schedule",
                raw_text=raw_text,
                source_start=schedule_start,
                source_end=schedule_end,
                page_starts=page_starts,
                part_number=None,
                part_title=None,
                article_number=None,
                article_title=None,
                schedule_number=schedule_number,
                schedule_title=schedule_title,
            )
        )

    return chunks


def validate_chunks(chunks: List[Dict[str, Any]]) -> None:
    """Fail loudly when output loses structure or contains invalid references."""
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    articles = [chunk for chunk in chunks if chunk["chunk_type"] == "article"]
    article_numbers = [chunk["metadata"]["article_number"] for chunk in articles]
    if article_numbers != sorted(article_numbers):
        raise ValueError("Article chunks are not in ascending document order.")

    chunk_id_set = set(chunk_ids)
    for chunk in chunks:
        if not chunk["text"].strip():
            raise ValueError(f"Empty chunk generated: {chunk['chunk_id']}")
        parent_id = chunk["parent_chunk_id"]
        if parent_id is not None and parent_id not in chunk_id_set:
            raise ValueError(f"Missing parent for chunk: {chunk['chunk_id']}")
        if chunk["metadata"]["page_start"] is None:
            raise ValueError(f"Missing page metadata: {chunk['chunk_id']}")

    print(
        "Validation: "
        f"{len(articles)} articles, "
        f"{sum(c['chunk_type'] == 'clause' for c in chunks)} clauses, "
        f"{len(chunks)} total chunks."
    )


def process_constitution_pdf(pdf_filepath: str, output_filepath: str) -> None:
    print(f"Reading PDF from: {pdf_filepath}...")
    raw_text, page_starts = extract_text_from_pdf(pdf_filepath)
    chunks = parse_constitution_to_json(raw_text, page_starts)
    validate_chunks(chunks)

    output_path = Path(output_filepath)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done! Created {len(chunks)} chunks in '{output_filepath}'.")


if __name__ == "__main__":
    process_constitution_pdf("const.pdf", "nepal_constitution_chunks.json")
