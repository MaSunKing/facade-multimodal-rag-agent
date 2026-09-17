"""Conservative CPU fact keys. Aliases normalise fields, never select tools.

Only explicit entity/value bindings are eligible. Unknown entity or incompatible
scope is not guessed from the question. Groups are disagreement candidates,
not a verdict about which document is correct.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

METRIC_ALIASES = {
    'thickness': ('thickness', '板厚', '厚度'),
    'width': ('width', '宽度', '板宽'),
    'length': ('length', '长度', '板长'),
    'weight': ('weight', 'mass', '重量', '质量'),
    'cost': ('cost', 'price', '单价', '价格', '成本'),
    'area': ('area', '面积'),
}
UNITS = {
    'mm': ('mm', Decimal('1')), '毫米': ('mm', Decimal('1')),
    'cm': ('mm', Decimal('10')), '厘米': ('mm', Decimal('10')),
    'm': ('mm', Decimal('1000')), '米': ('mm', Decimal('1000')),
    'kg': ('kg', Decimal('1')), '千克': ('kg', Decimal('1')),
    'g': ('kg', Decimal('0.001')), '克': ('kg', Decimal('0.001')),
    'm²': ('m²', Decimal('1')), 'm2': ('m²', Decimal('1')), '㎡': ('m²', Decimal('1')),
    '元': ('CNY', Decimal('1')), '万元': ('CNY', Decimal('10000')),
    '亿元': ('CNY', Decimal('100000000')), 'cny': ('CNY', Decimal('1')),
    'usd': ('USD', Decimal('1')), '%': ('%', Decimal('1')),
}
_aliases = sorted((a for aliases in METRIC_ALIASES.values() for a in aliases), key=len, reverse=True)
_units = sorted(UNITS, key=len, reverse=True)
_FACT = re.compile(
    r'(?P<metric>' + '|'.join(map(re.escape, _aliases)) + r')'
    r"\s*(?:为|是|[:：=]|\bis\b)?\s*['\"]?"
    r'(?P<value>-?\d+(?:,\d{3})*(?:\.\d+)?)\s*'
    r'(?P<unit>' + '|'.join(map(re.escape, _units)) + r')(?![A-Za-z])', re.I)


def canonical_metric(value: str) -> str:
    folded = value.strip().casefold()
    return next((key for key, names in METRIC_ALIASES.items() if folded in names), folded)


def extract_facts(item: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(item.get('text') or '')
    facts = []
    headers = {}
    for header in re.findall(r'(?m)^\[COLUMNS\]\s*([^\n]*)', text):
        for column, label in re.findall(r'(?:^|\|)\s*([A-Z]+)=([^|]+)', header):
            headers[column] = label.strip()
    for line in text.splitlines():
        # Strip only recognised parser wrappers from the fact-matching view.
        # Keep canonical text unchanged; an arbitrary bracketed prefix is NOT
        # evidence of an entity and must remain ineligible.
        line = re.sub(r'^\s*\[BLOCK\s+id=[^\n]*?\]\s*', '', line)
        cells = [(label or headers.get(column, ''), value)
                 for column, label, value in re.findall(r"\b([A-Z]+)\d+(?:\[([^\]]+)\])?='([^']*)'", line)]
        row = {key.strip().casefold(): value for key, value in cells}
        row_entity = next((row[k] for k in ('entity', 'product', 'product_name', 'model', '产品', '产品名称', '品名', '型号') if row.get(k)), None)
        if row_entity:
            scope = next((row[k] for k in ('period', 'year', '期间', '年份') if row.get(k)), item.get('fact_scope', ''))
            version = next((row[k] for k in ('version', 'source version', 'date', '版本', '日期') if row.get(k)), item.get('version', ''))
            for key, value in row.items():
                metric = canonical_metric(key)
                if metric in METRIC_ALIASES:
                    if re.fullmatch(r'-?\d+(?:\.\d+)?', value) and row.get('unit'):
                        value += row['unit']
                    facts.extend(extract_facts({**item, 'text':f'{metric}: {value}',
                        'entity':row_entity, 'fact_scope':scope, 'version':version}))
        for match in _FACT.finditer(line):
            prefix = line[:match.start()].strip(' \t:：|,，')
            # Parser wrappers and unbound prose are not entity IDs. Native
            # table row context can supply an explicit entity instead.
            context = item.get('row_context') or {}
            entity = item.get('entity') or context.get('entity') or context.get('product_name')
            if not entity and prefix and not prefix.startswith('[') and len(prefix) <= 80:
                entity = re.sub(r'\s+(?:has|的)$', '', prefix, flags=re.I)
            if not entity:
                continue
            unit, factor = UNITS[match.group('unit').casefold()]
            try:
                value = Decimal(match.group('value').replace(',', '')) * factor
            except InvalidOperation:
                continue
            facts.append(dict(entity=re.sub(r'\s+', '', str(entity)).casefold(),
                metric=canonical_metric(match.group('metric')), value=str(value.normalize()), unit=unit,
                scope=str(item.get('fact_scope') or item.get('period') or ''),
                version=str(item.get('version') or item.get('effective_date') or ''),
                evidence_id=str(item.get('evidence_id') or ''),
                source_id=str(item.get('document_id') or item.get('document_name') or item.get('source_url') or ''),
                provenance='explicit_text_binding_not_verified_fact'))
    return facts


def disagreement_groups(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[dict]] = {}
    for item in evidence:
        if item.get('facts_eligible') is False:
            continue
        for fact in extract_facts(item):
            key = tuple(fact[k] for k in ('entity', 'metric', 'scope', 'version', 'unit'))
            buckets.setdefault(key, []).append(fact)
    groups = []
    for key, facts in buckets.items():
        ids = list(dict.fromkeys(f['evidence_id'] for f in facts))
        if len(ids) < 2 or len({f['value'] for f in facts}) < 2:
            continue
        # Two values in one source may be unrelated measurements; do not
        # label this a cross-document conflict without a distinct source.
        known = {f['source_id'] for f in facts if f['source_id']}
        if all(f['source_id'] for f in facts) and len(known) < 2:
            continue
        groups.append(dict(entity=key[0], metric=key[1], scope=key[2], version=key[3], unit=key[4],
            evidence_ids=ids, values=list(dict.fromkeys(f['value'] for f in facts)),
            source_identity_verified=all(f['source_id'] for f in facts),
            status='potential_disagreement_requires_source_scope_review'))
    return groups
