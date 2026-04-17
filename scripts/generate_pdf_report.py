from __future__ import annotations

import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "report" / "report.md"
TARGET = ROOT / "report" / "agent_rh_report.pdf"


def normalize_markdown(markdown_text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        for marker in ("# ", "## ", "### ", "- ", "`"):
            if line.startswith(marker):
                line = line.removeprefix(marker).strip()
        line = line.replace("`", "")
        wrapped = textwrap.wrap(line, width=92) or [""]
        lines.extend(wrapped)
    return lines


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_page_stream(page_lines: list[str]) -> bytes:
    commands = ["BT", "/F1 10 Tf", "40 800 Td", "14 TL"]
    for line in page_lines:
        safe = pdf_escape(line)
        commands.append(f"({safe}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", errors="replace")


def assemble_pdf(page_streams: list[bytes]) -> bytes:
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_object_ids = []
    first_page_id = 4
    for index in range(len(page_streams)):
        page_object_ids.append(first_page_id + index * 2)

    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_object_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, stream in enumerate(page_streams):
        page_id = first_page_id + index * 2
        stream_id = page_id + 1
        page_object = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>"
        ).encode("latin-1")
        stream_object = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        objects.append(page_object)
        objects.append(stream_object)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)


def main() -> None:
    lines = normalize_markdown(SOURCE.read_text(encoding="utf-8"))
    lines_per_page = 48
    page_streams = []
    for index in range(0, len(lines), lines_per_page):
        page_streams.append(build_page_stream(lines[index : index + lines_per_page]))
    TARGET.write_bytes(assemble_pdf(page_streams))
    print(f"PDF generated: {TARGET}")


if __name__ == "__main__":
    main()
