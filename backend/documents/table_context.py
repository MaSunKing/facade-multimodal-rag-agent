"""Native table header relations across parser chunks, without inferred units.

Every context string is an original cell assignment. Merged ranges define
column scope; a new full-width heading resets the previous section. These
are source-context candidates, not verified business facts.
"""
from __future__ import annotations
import re
from typing import Any

CELL = re.compile(r"([A-Z]{1,3})(\d+)(?:\[[^\]]*\])?='([^']*)'")
UNIT_HINT = re.compile(r'unit|单位|毫米|厘米|千克|平方米|万元|(?<![A-Za-z])(?:mm|cm|m|kg|g|kWh|MWh|megawatthours|tons?|barrels?|dollars?|cents?|USD|EUR|CNY|hours?|days?)(?![A-Za-z])', re.I)


def column_number(label: str) -> int:
    result = 0
    for char in label:
        result = result * 26 + ord(char) - ord('A') + 1
    return result


def native_table_contexts(chunks: list[dict[str, Any]]) -> dict[str, dict[int, dict]]:
    tables: dict[str, dict] = {}
    for chunk in chunks:
        text = str(chunk.get('text') or '')
        table = re.match(r'\[TABLE id=([^\n]*?) sheet=.*? range=([A-Z]+)\d+:([A-Z]+)\d+', text)
        if not table:
            continue
        state = tables.setdefault(table[1], dict(rows={}, merged={}, width=column_number(table[3])))
        for a, row, b, end in re.findall(r'\b([A-Z]+)(\d+):([A-Z]+)(\d+)\b',
                                        '\n'.join(re.findall(r'(?m)^\[MERGED_RANGES[^\n]*', text))):
            if row == end:
                state['merged'][(a, int(row))] = column_number(b)
        for line in re.findall(r'(?m)^\[ROW[^\n]*', text):
            cells = list(CELL.finditer(line))
            if cells:
                state['rows'][int(cells[0][2])] = cells
    output = {}
    for table_id, state in tables.items():
        headers: dict[int, list[str]] = {}
        section: list[str] = []
        contexts = {}
        for row, cells in sorted(state['rows'].items()):
            data_row = any(re.fullmatch(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?\s*(?:%|mm|cm|m|kg|g|元|万元|毫米|厘米)?', cell[3].strip(), re.I) for cell in cells)
            if not data_row:
                # Share immutable header snapshots across data rows; copying
                # every column for every row bloats large workbook requests.
                headers = {k:list(v) for k,v in headers.items()}
                for cell in cells:
                    value = cell[3].strip().replace('\\n', ' ').strip()
                    if not value or len(value) > 220:
                        continue
                    start = column_number(cell[1])
                    end = state['merged'].get((cell[1], row), start)
                    if start == 1 and end == state['width'] and state['width'] > 1:
                        section = [cell[0]]
                        # Units from the preceding section cannot leak through
                        # a new full-width section boundary. Column headings
                        # such as years remain useful, but are not unit facts.
                        headers = {col: [s for s in values if not UNIT_HINT.search(s)]
                                   for col, values in headers.items()}
                    else:
                        for col in range(start, end + 1):
                            previous = headers.setdefault(col, [])
                            if cell[0] not in previous:
                                previous.append(cell[0])
                            headers[col] = previous[-3:]
            contexts[row] = dict(section=section, columns=headers)
        output[table_id] = contexts
    return output


def row_context_cells(context: dict, columns: list[str]) -> list[str]:
    return list(dict.fromkeys([*context.get('section', []),
        *(cell for col in columns for cell in context.get('columns', {}).get(column_number(col), []))]))
