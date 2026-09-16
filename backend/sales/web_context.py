"""Compact public evidence, preserving dates and same-publisher distinctions."""
import re
import html
from urllib.parse import urlsplit


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


def prepare_web_context(sources, target_date):
    output=[]; exact={}; groups={}; removed=[]
    for source in sources:
        text=clean_excerpt(source.get('excerpt') or source.get('search_excerpt') or '')
        if not text: continue
        group=publisher_group(str(source.get('url') or ''))
        key=(group,re.sub(r'\s+',' ',text))
        source_id=str(source.get('source_id') or '')
        if key in exact:
            exact[key]['duplicate_source_ids'].append(source_id); removed.append(source_id); continue
        item={'source_id':source_id,'title':str(source.get('title') or ''),
              'publisher_group':group,'excerpt':text,
              'source_date_metadata':str(source.get('date') or ''),
              'evidence_level':str(source.get('evidence_level') or 'unverified_search_excerpt'),
              'mentioned_dates':list(dict.fromkeys(re.findall(r'\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',text)))[:16],
              'duplicate_source_ids':[]}
        exact[key]=item; output.append(item); groups.setdefault(group,[]).append(source_id)
    return output, {'raw_source_count':len(sources),'kept_source_count':len(output),
                    'exact_duplicates_removed':removed,'same_publisher_groups':{k:v for k,v in groups.items() if len(v)>1},
                    'target_date':target_date,
                    'policy':'Match the requested absolute date and place, not a page-relative today. '
                    'Same-publisher pages are not independent confirmations. Keep differing dated versions as potential conflicts. '
                    'Search timestamps do not prove the observation or forecast date. Never merge conflicting values into one fact.'}
