"""CPU-only extraction and bounded query windows; no generated page facts."""
import re
from html import unescape


def extract_page_text(content: str):
    if '<' not in content or '>' not in content:
        return content.strip(), 'plain_text'
    try:
        import trafilatura
        text = trafilatura.extract(content, include_comments=False, include_tables=True,
                                   include_links=False, favor_recall=True, deduplicate=True)
        if text and len(text.strip()) >= 80:
            return text.strip(), 'trafilatura'
    except ImportError:
        pass
    except Exception:
        pass
    # Structure-aware fallback, preserving table labels/values and short clauses.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    for node in soup.select('script,style,noscript,svg,nav,footer,header,aside,form'):
        node.decompose()
    root = soup.find('article') or soup.find('main') or soup.body or soup
    for row in root.find_all('tr'):
        values = [cell.get_text(' ', strip=True) for cell in row.find_all(['th','td'])]
        row.replace_with('\n' + ' | '.join(values) + '\n')
    blocks = []
    for line in root.get_text('\n', strip=True).splitlines():
        line = re.sub(r'[ \t]+', ' ', unescape(line)).strip()
        if line:
            blocks.append(line)
    return '\n'.join(blocks), 'structural_html_fallback'


def select_page_windows(text: str, query: str, limit: int = 1800):
    # Original parsed text remains cached; this is a question-specific view.
    from backend.sales.context_engine import _terms
    terms = _terms(query)
    windows = []
    for paragraph in re.split(r'\n\s*\n|\n', text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Long paragraphs get overlapping windows; late answers stay eligible.
        for start in range(0, len(paragraph), 480):
            part = paragraph[start:start+600]
            windows.append(part)
            if start+600 >= len(paragraph):
                break
    scored = sorted(enumerate(windows), key=lambda pair: (
        -len(terms & _terms(pair[1])), pair[0]))
    selected = []
    used = 0
    seen = set()
    for index, window in scored:
        if window in seen or used+len(window)+1 > limit:
            continue
        selected.append((index, window)); seen.add(window); used += len(window)+1
    # Restore source order, not a misleading collage ordered by keyword score.
    return '\n'.join(window for _,window in sorted(selected)), {
        'parsed_characters':len(text), 'candidate_window_count':len(windows),
        'selected_window_indices':[i for i,_ in sorted(selected)],
        'selected_characters':used, 'policy':'query_windows_not_prefix_truncation'}
