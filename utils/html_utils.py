from typing import List

from bs4 import NavigableString, Tag


def parse_html_to_text(bs4_node: Tag) -> str:
    """
    Parse a BeautifulSoup node into formatted text,
    preserving structure and avoiding extraneous newlines.
    """
    lines: List[str] = []
    current_line = ""
    for child in bs4_node.children:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                current_line += text + " "
        elif isinstance(child, Tag):
            if child.name == "p":
                if current_line.strip():
                    lines.append(current_line.strip())
                    current_line = ""
                para = child.get_text(" ", strip=True)
                if para:
                    lines.append(para)
            elif child.name in ("ul", "ol"):
                if current_line.strip():
                    lines.append(current_line.strip())
                    current_line = ""
                for li in child.find_all("li"):
                    li_text = li.get_text(" ", strip=True)
                    if li_text:
                        lines.append(f"- {li_text}")
            elif child.name == "br":
                if current_line.strip():
                    lines.append(current_line.strip())
                    current_line = ""
                lines.append("")
            else:
                inline = child.get_text(" ", strip=True)
                if inline:
                    current_line += inline + " "
    if current_line.strip():
        lines.append(current_line.strip())
    cleaned: List[str] = []
    prev_blank = False
    for line in lines:
        if not line and not prev_blank:
            cleaned.append("")
            prev_blank = True
        elif line:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()
