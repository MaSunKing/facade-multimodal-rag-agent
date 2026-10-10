"""Protect explicit/relative time anchors without choosing tools or topics."""
import re
from datetime import date, timedelta


def resolve_query_time(query: str, current_date: str) -> str:
    """Resolve literal relative days; never infer source fact validity."""
    value = query.strip()
    anchor = date.fromisoformat(current_date)
    offsets = {'今天': 0, '今日': 0, '本日': 0, '昨天': -1, '昨日': -1,
               '明天': 1, '明日': 1, 'today': 0, 'yesterday': -1, 'tomorrow': 1}
    pattern = r'今天|今日|本日|昨天|昨日|明天|明日|\b(?:today|yesterday|tomorrow)\b'
    value = re.sub(pattern, lambda m: (anchor + timedelta(days=offsets[m.group().lower()])).isoformat(),
                   value, flags=re.I)
    if re.search(r'现在|当前|最新|\b(?:current|latest|now)\b', value, re.I):
        value += f' 当前日期 {current_date}'
    return value


def mentioned_dates(text: str) -> list[str]:
    """Literal mentions, not classified fact dates; don't invent missing years."""
    pattern = (r'(?<!\d)(?:19|20)\d{2}(?:[-/]\d{1,2}[-/]\d{1,2}|年\d{1,2}月\d{1,2}日?)'
               r'|(?<![\d/-])\d{1,2}月\d{1,2}日'
               r'|(?<![\d/-])\d{1,2}/\d{1,2}(?![\d/-])')
    return list(dict.fromkeys(re.findall(pattern, text)))[:16]


def protect_query_time(question: str, query: str, current_date: str):
    explicit = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', question))
    current = bool(re.search(r'现在|当前|最新|今天|今日|\b(?:current|latest|today|now)\b', question, re.I))
    allowed = explicit | ({current_date[:4]} if current else set())
    invented = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', query)) - allowed
    if allowed and invented:
        # Reject the rewrite rather than guessing how to repair its meaning.
        return question, {'applied': True, 'reason': 'rewrite_changed_time_anchor',
                          'rejected_years': sorted(invented), 'used_original_question': True}
    return query, {'applied': False}
