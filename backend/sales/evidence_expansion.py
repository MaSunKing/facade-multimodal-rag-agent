"""CPU-only native-source expansion. Structure checks are not semantic verdicts."""
import hashlib
import re


def source_body(text):
    return re.sub(r'^资料角色：[^\n]*\n', '', str(text or '')).strip()


def heading_only(row):
    body = source_body(row.get('text'))
    return bool(body and len(body) <= 60 and '\n' not in body
                and not re.search(r'[。！？.!?；;：:]|\d+\s*(?:mm|kg|%|℃)', body)
                and (row.get('section_heading') == body or
                     re.match(r'^(?:[一二三四五六七八九十]+[、.]|\d+[.、]\s*|第.+[章节])', body)))


def needs_expansion(row):
    if (row.get('facts_eligible') is False or row.get('directory_collection')
            or row.get('catalog_record_type') or row.get('visual_description')
            or row.get('source_type') in {'visual', 'catalog_labels'}):
        return None
    if heading_only(row):
        return 'heading_without_body'
    if row.get('preview_truncated') or row.get('windowed'):
        return 'bounded_excerpt'
    return None


def complete_blocks(text):
    """Never slice steps/tables/numeric conditions at arbitrary punctuation."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', str(text)) if p.strip()]
    for paragraph in paragraphs:
        if (re.search(r'^\s*(?:\d+[.、)]|[一二三四五六七八九十]+[、.]|\[ROW|\[TABLE)', paragraph)
                or '|' in paragraph or '[COLUMNS]' in paragraph):
            yield paragraph
        else:
            # Sentence ends, not commas/decimal points. Unterminated tails
            # remain whole native blocks; never cut a condition in half.
            yield from [s.strip() for s in re.split(r'(?<=[。！？!?])\s*', paragraph) if s.strip()]


def expand_with_budget(rows, *, provider, token_count, token_limit,
                       max_chars=10000, per_evidence_chars=2000, secondary_tokens=None, secondary_limit=None):
    """Fit the original Context first, then tentatively add complete blocks.

    Provider is lazy and owner/ACL scoped. No network, pixels or extra model
    calls. Failed additions roll back without deleting original evidence.
    """
    selected = [dict(row) for row in rows]
    def secondary_fits():
        return secondary_tokens is None or secondary_tokens(selected) <= secondary_limit
    initial = token_count(selected)
    audit = {'policy':'budget_first_native_blocks_v1', 'initial_prompt_tokens':initial,
             'token_limit':token_limit, 'character_limit':max_chars,
             'per_evidence_character_limit':per_evidence_chars, 'decisions':[],
             'semantic_support_verified':False, 'model_calls':0}
    seen = {source_body(r.get('text')) for r in selected}
    for index, row in enumerate(selected):
        reason = needs_expansion(row)
        if not reason:
            continue
        state = {'evidence_id':row['evidence_id'], 'trigger':reason,
                 'status':'not_expanded', 'added_chars':0, 'source_chunk_ids':[]}
        # This status is an explicit generation limitation, not source facts.
        row['expansion_status'] = {'status':'not_expanded', 'reason':reason,
                                  'body_complete':False}
        if token_count(selected) >= token_limit or sum(len(str(r.get('text',''))) for r in selected) >= max_chars:
            state['reason'] = 'budget_insufficient'
        else:
            candidates = provider(row)
            stop = False
            for candidate in candidates:
                for block in complete_blocks(candidate.get('text', '')):
                    if source_body(block) in seen or block in str(row.get('text','')):
                        continue
                    previous = dict(row)
                    row['text'] = str(previous.get('text','')).rstrip() + '\n\n' + block
                    row['citations'] = list(previous.get('citations', []))
                    for citation in candidate.get('citations', []):
                        if citation not in row['citations']:
                            row['citations'].append(citation)
                    row['expansion_status'] = {'status':'expanded', 'reason':reason,
                        'body_complete':False, 'boundary':'native_blocks_not_semantic_certification'}
                    if (len(row['text']) > per_evidence_chars or
                            sum(len(str(r.get('text',''))) for r in selected) > max_chars or
                            token_count(selected) > token_limit or not secondary_fits()):
                        row.clear(); row.update(previous)
                        state['reason'] = 'budget_boundary_rollback'
                        stop = True
                        break
                    seen.add(source_body(block))
                    state['added_chars'] += len(block)
                    if candidate.get('chunk_id') not in state['source_chunk_ids']:
                        state['source_chunk_ids'].append(candidate.get('chunk_id'))
                if stop:
                    break
            if state['added_chars']:
                state['status'] = 'expanded'
                state.setdefault('reason', 'native_boundary_reached')
            else:
                state.setdefault('reason', 'native_body_unavailable')
        row['expansion_status'].update(status=state['status'], reason=state['reason'],
                                      body_complete=False)
        row['prepared_content_sha256'] = hashlib.sha256(str(row['text']).encode('utf-8')).hexdigest()
        # Even diagnostic metadata must fit the complete wire budget. The
        # unchanged heading remains visibly insufficient if no status fits.
        if token_count(selected) > token_limit or not secondary_fits():
            row.clear(); row.update(rows[index])
            state.update(status='not_expanded',reason='status_budget_rollback',added_chars=0,
                         source_chunk_ids=[])
        audit['decisions'].append(state)
    audit['final_prompt_tokens'] = token_count(selected)
    audit['budget_satisfied'] = audit['final_prompt_tokens'] <= token_limit and secondary_fits()
    return selected, audit


def native_provider(sources, *, company_documents=(), customer_documents=()):
    """Only use admitted sources and already-authorized native text assets."""
    company = (sources.get('company_rag') or {}).get('text_evidence', [])
    anchors = {str(r.get('evidence_id') or f'T{i}'):r for i,r in enumerate(company,1)}
    def provide(row):
        raw = anchors.get(row['evidence_id'])
        if raw:
            anchor_id = raw.get('chunk_id')
            original = next((d for d in company_documents if d.get('id') == anchor_id), None)
            if not original:
                return []
            ref = (original.get('source_refs') or [{}])[0]
            identity = ref.get('document_id') or ref.get('document_name')
            page = ref.get('source_page')
            peers = []
            for d in company_documents:
                refs = d.get('source_refs') or []
                for source in refs:
                    if (d.get('kind') == 'text' and identity and
                            (source.get('document_id') or source.get('document_name')) == identity and
                            (page is None or (source.get('source_page') is not None and
                                             page <= source['source_page'] <= page+1))):
                        peers.append((d,source)); break
            peers.sort(key=lambda pair:(pair[1].get('source_page') or 0,
                (pair[1].get('bbox') or [0,0])[1]))
            position = next((i for i,(d,_) in enumerate(peers) if d.get('id') == anchor_id), None)
            if position is None:
                return []
            output = []
            for d,source in peers[position+1:position+17]:
                if heading_only({'text':d.get('text'), 'section_heading':source.get('section_heading')}):
                    # A numbered subsection belongs to its parent section;
                    # next chapter stops expansion, not the next step label.
                    if re.match(r'^[一二三四五六七八九十]+[、.]', str(d.get('text',''))):
                        break
                public_ref = {k:source.get(k) for k in ('document_name','source_page','section_heading',
                    'bbox','source_format','source_sheet','source_range','source_url')}
                output.append({'chunk_id':d['id'], 'text':d.get('text',''), 'citations':[public_ref]})
            return output
        # Restore only the selected customer's native chunk, not a different
        # file or an unselected session. Ownership checked by caller.
        for doc in customer_documents:
            if row.get('document_id') != doc.document_id:
                continue
            chunk_id = row.get('original_chunk_id') or row.get('chunk_id')
            chunk = next((c for c in doc.chunks if c.get('chunk_id') == chunk_id), None)
            if chunk:
                return [{'chunk_id':chunk_id,'text':chunk.get('text',''),
                         'citations':row.get('citations',[])}]
        # A search snippet never grants permission to fetch or fabricate a
        # page. Existing fetched text is reusable only from this exact URL.
        web = sources.get('public_web_search')
        for source in web[0] if isinstance(web,(tuple,list)) and len(web)==2 else []:
            if row['evidence_id'] == source.get('source_id') and source.get('page_fetch_status') in {'fetched','success'}:
                return [{'chunk_id':source['source_id'],'text':source.get('page_text') or source.get('body_text') or '',
                         'citations':row.get('citations',[])}]
        return []
    return provide
