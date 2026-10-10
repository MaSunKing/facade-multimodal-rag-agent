"""Compact public evidence, preserving dates and same-publisher distinctions."""
import re
import html
from urllib.parse import urlsplit
from backend.sales.query_constraints import mentioned_dates


def web_evidence_row(source):
    """Carry provenance into ranking/audit; page dates are not fact dates."""
    text = clean_excerpt(source.get('excerpt') or source.get('search_excerpt') or '')
    return {'evidence_id': source['source_id'], 'text': text,
            'requirement_ids':source.get('requirement_ids', []),
            'retrieval_requirements':source.get('retrieval_requirements', []),
            'source_type': 'public_web', 'document_name': source.get('title'),
            'source_url': source.get('url'), 'publisher_group': publisher_group(source.get('url') or ''),
            'source_date': source.get('date'), 'date_scope': 'page_metadata_not_fact_date',
            'mentioned_dates': mentioned_dates(text),
            'evidence_level': source.get('evidence_level'),
            'page_fetched_at_china': source.get('page_fetched_at_china'),
            'web_original_rank': int(source.get('original_rank') or 0),
            'page_fetch_status': source.get('page_fetch_status')}


def clean_excerpt(value):
    # Preserve table/paragraph boundaries before removing HTML decorations.
    value=re.sub(r'</(?:td|th)>',' | ',str(value),flags=re.I)
    value=re.sub(r'</(?:tr|p|div)>','\n',value,flags=re.I)
    return re.sub(r'[ \t]+',' ',html.unescape(re.sub(r'<[^>]+>',' ',value))).strip()


def publisher_group(url):
    host=(urlsplit(url).hostname or '').lower().rstrip('.')
    parts=host.split('.')
    suffix='.'.join(parts[-2:])
    # Grouping only, never used as an allow-list or proof of authority.
    count=3 if suffix in {'com.cn','gov.cn','org.cn','net.cn','co.uk','gov.uk','org.uk','com.au'} else 2
    return '.'.join(parts[-count:])


def pack_web_context(sources, tokenizer, max_tokens=4000):
    """Bound actual web TEXT tokens, not characters; metadata budget is separate."""
    raw_tokens = sum(len(tokenizer.encode(s['excerpt'],add_special_tokens=False)) for s in sources)
    kept = []
    used = 0
    omitted = []
    for source in sources:
        selected = []
        for block in source['excerpt'].splitlines():
            for start in range(0,len(block),500):
                window = block[start:start+500]
                size = len(tokenizer.encode(window+'\n',add_special_tokens=False))
                if used+size <= max_tokens:
                    selected.append(window); used += size
        if selected:
            kept.append({**source,'excerpt':'\n'.join(selected)})
        else:
            omitted.append(source['source_id'])
    return kept, {'raw_web_text_tokens':raw_tokens,'kept_web_text_tokens':used,
                  'web_text_token_limit':max_tokens,'omitted_source_ids':omitted,
                  'kept_source_ids':[s['source_id'] for s in kept]}


def prepare_web_context(sources, target_date):
    output=[]; exact={}; groups={}; removed=[]
    for source in sources:
        text=clean_excerpt(source.get('excerpt') or source.get('search_excerpt') or '')
        if not text: continue
        group=publisher_group(str(source.get('url') or ''))
        key=(group,re.sub(r'\s+',' ',text))
        source_id=str(source.get('source_id') or '')
        if key in exact:
            from backend.sales.requirements import merge_requirements
            merge_requirements(exact[key], source)
            exact[key]['duplicate_source_ids'].append(source_id); removed.append(source_id); continue
        item={'source_id':source_id,'title':str(source.get('title') or ''),
              'requirement_ids':source.get('requirement_ids', []),
              'retrieval_requirements':source.get('retrieval_requirements', []),
              'publisher_group':group,'excerpt':text,
              'source_date_metadata':str(source.get('date') or ''),
              'evidence_level':str(source.get('evidence_level') or 'unverified_search_excerpt'),
              'page_fetched_at_china':source.get('page_fetched_at_china'),
              'mentioned_dates':mentioned_dates(text),
              'duplicate_source_ids':[]}
        exact[key]=item; output.append(item); groups.setdefault(group,[]).append(source_id)
    return output, {'raw_source_count':len(sources),'kept_source_count':len(output),
                    'exact_duplicates_removed':removed,'same_publisher_groups':{k:v for k,v in groups.items() if len(v)>1},
                    'target_date':target_date,
                    'policy':'Match the requested absolute date and place, not a page-relative today. '
                    'Same-publisher pages are not independent confirmations. Keep differing dated versions as potential conflicts. '
                    'Search timestamps do not prove the observation or forecast date. Never merge conflicting values into one fact.'}
