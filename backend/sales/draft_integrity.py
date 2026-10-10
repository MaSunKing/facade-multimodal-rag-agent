"""Source-independent delivery guards; no fact matching or new tool calls."""
import re
from collections import Counter

def canonical_listing(reply, rows, *, action='none', aspects=()):
    """Render known directory names once, only in a name-list task.

    This is presentation of backend-owned membership, like source cards, not
    fact-string validation. Do not alter technical steps or comparison values.
    Unknown/ambiguous names stay untouched for task review, never guessed.
    """
    if action != 'list' or aspects:
        return reply, {'applied':False}
    from backend.sales.relation_scope import normalized
    members = {m['id']:m for row in rows if row.get('directory_collection')
               and row.get('catalog_record_type') != 'directory_aggregate'
               for m in row.get('directory_members', [])}
    if not members: return reply, {'applied':False}
    output, seen, removed, canonicalized = [], set(), 0, 0
    fenced = False
    for line in reply.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        match = None if fenced else re.match(r'^(\s*(?:[-*•]\s+|\d+[.)、]\s*))(.+)$',line)
        if not match:
            if line.strip(): seen.clear()  # Separate sections may share a referent.
            output.append(line); continue
        text = normalized(match[2])
        candidates = [m for m in members.values() if len(normalized(m.get('name',''))) >= 3
                      and normalized(m['name']) in text]
        if len(candidates) != 1:
            output.append(line); continue
        member = candidates[0]
        if member['id'] in seen:
            removed += 1; continue
        seen.add(member['id'])
        output.append(match[1]+member['name'])
        canonicalized += 1
    return '\n'.join(output), {'applied':bool(canonicalized or removed),
        'policy':'backend_directory_identity_name_list_only',
        'duplicate_items_removed':removed,'canonical_items_rendered':canonicalized,
        'new_facts_added':False}

def repeated_cycle(reply):
    """Detect long exact output loops, not shared values across distinct rows.

    Ignore fenced examples. Require three adjacent multi-line copies and a
    substantial period: common headings/numbers alone are never an error.
    """
    reply = re.sub(r'```.*?```', '', reply, flags=re.S)
    # A list can repeat without forming an adjacent periodic cycle. Scope this
    # check to one uninterrupted list section: equal values under A/B headings
    # are legitimate comparison content, not duplicates of one delivery item.
    groups = [[]]
    for line in reply.splitlines():
        if not line.strip(): continue
        match = re.match(r'^\s*(?:[-*•]\s+|\d+[.)、]\s*)(.+)$', line)
        if match:
            groups[-1].append(re.sub(r'\s+',' ',match[1]).strip())
        elif groups[-1]:
            groups.append([])
    for group in groups:
        counts = Counter(group)
        repeated = [n for n in counts.values() if n > 1]
        if len(repeated) >= 3 and sum(n-1 for n in repeated) >= 6:
            return {'kind':'duplicate_list_entries_in_same_section',
                    'duplicate_entry_count':sum(n-1 for n in repeated),
                    'distinct_repeated_entries':len(repeated), 'fact_support_verified':False}
    lines = [re.sub(r'^\s*(?:[-*•]|\d+[.)、])\s*', '', line).strip()
             for line in reply.splitlines() if line.strip()]
    for start in range(len(lines)):
        for size in range(3, min(40,(len(lines)-start)//3)+1):
            block = lines[start:start+size]
            if sum(map(len,block)) < 240: continue
            if (block == lines[start+size:start+2*size]
                    and block == lines[start+2*size:start+3*size]):
                return {'kind':'exact_repeated_delivery_cycle','period_lines':size,
                        'minimum_copies':3,'fact_support_verified':False}
    return None

def guard_review(review, reply, question):
    """Legacy adapter only: never override the semantic review by counts."""
    return review
