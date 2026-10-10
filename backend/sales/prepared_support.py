"""Question-sized evidence before the semantic gate; canonical sources stay intact."""
import hashlib
import json
from urllib.parse import urlsplit
from backend.sales.evidence_gate import terms
from backend.sales.evidence_packing import compact_source_text


def fit_support_budget(rows, *, audit_tokens, generation_tokens, ceiling, audit_limit=None, final_limit=None, compact_overview=None, question=None, goals=(), target_tokens=4500, required_document_ids=(), scope='unknown', operation='unknown', retrieval_requirements=(), collection_action='auto'):
    """Select whole groups before audit, preserving files and required goals.

    Both counters include their system/wire envelope; generation must also
    reserve space for verdict/visual fields only available after the audit.
    """
    selected = list(rows)
    directory_original = list(rows)
    compression = {}
    original_audit_tokens = audit_tokens(selected)
    audit_limit = ceiling if audit_limit is None else audit_limit
    final_limit = ceiling if final_limit is None else final_limit
    originally_fits = original_audit_tokens <= audit_limit and generation_tokens(selected) <= final_limit
    # Summary coverage comes from actual directory packets, not a second
    # optional Planner field. Count/list legacy requests retain their policy.
    if operation=='summary' and any(r.get('directory_collection') for r in rows) and collection_action=='none':
        collection_action='auto'
    directory_compaction = False
    if not originally_fits and collection_action != 'none':
        from backend.sales.directory_evidence import compact_directory_packets
        compacted_rows = compact_directory_packets(selected, question or '', goals)
        directory_compaction = compacted_rows != selected
        selected = compacted_rows
    if question is not None and not originally_fits:
        from backend.sales.support_compression import compress_support
        selected, compression = compress_support(selected, question, goals)
    from backend.sales.requirements import representatives, coverage
    requirements = list({r['requirement_id']:r for r in [*retrieval_requirements,
        *[v for row in selected for v in row.get('retrieval_requirements', [])]]}.values())
    # Keep one representative per retrieval cell when possible, not every
    # candidate. Missing cells stay observable, never fabricated as supported.
    # Reuse upstream semantic scores without another inference call. Ordering
    # is a selection priority, never proof of support. Preserve input verbatim
    # when both complete prompts already fit.
    if not originally_fits:
        selected.sort(key=lambda r:max((r.get('goal_support_scores') or {}).values(),default=0),reverse=True)
    reps = representatives(selected) if not originally_fits else []
    rep_ids = {r['evidence_id'] for r in reps}
    selected = [*reps, *(r for r in selected if r['evidence_id'] not in rep_ids)]
    # Reserve real content by backend source/file identity, not gallery counts.
    # Relevance scores nominate candidates, never certify semantic support.
    reserved = set()
    buckets = {}
    required = set(required_document_ids)
    for row in selected:
        if row.get('gallery_collection') and row.get('evidence_scope') == 'display_labels':
            reserved.add(row['evidence_id'])  # One bounded display manifest, not every image.
        if row.get('directory_collection'):
            continue  # Reserve one compact packet per category below, not every packet.
        if (row.get('catalog_record_type') == 'product_collection' and row.get('entity_coverage_ids')
                and collection_action != 'none'):
            reserved.add(row['evidence_id'])  # A compact SQL member list represents the selected set.
        if row.get('visual_description') and row.get('document_id') in required:
            reserved.add(row['evidence_id'])
        if (row.get('source_type') != 'visual' and not row.get('visual_description')
                and row.get('facts_eligible') is not False and str(row.get('text') or '').strip()
                and row.get('evidence_scope') not in {'display_labels','document_index'}):
            identity = row.get('document_id') or row.get('document_name') or ''
            # Explicit uploaded targets retain a representative per file.
            # Incidental company/web documents are candidates, not mandatory
            # targets merely because they have different provenance names.
            key = (row.get('source_type') or 'text',
                   identity if question is None or row.get('document_id') in required
                   or row.get('required_document_coverage') else '')
            buckets.setdefault(key,[]).append(row)
    hard_reserved = set(reserved)
    for members in buckets.values():
        collections = [r for r in members if r.get('catalog_record_type') == 'product_collection'
                       and r.get('entity_coverage_ids') and collection_action != 'none']
        winner = max(collections or members,key=lambda row:max((row.get('goal_support_scores') or {}).values(),default=0))
        reserved.add(winner['evidence_id'])
    directory_groups = {}
    for row in selected:
        if row.get('directory_collection'):
            directory_groups.setdefault(row['directory_category'], []).append(row)
    # Lists/counts need membership packets; an installation explanation needs
    # source content, not a compulsory unrelated directory of every chapter.
    if collection_action != 'none':
        for members in directory_groups.values():
            reserved.add(members[0]['evidence_id'])
            hard_reserved.add(members[0]['evidence_id'])
    removed = []
    compacted = []
    desired_audit = audit_limit
    directory_names_attempted = False
    while audit_tokens(selected) > desired_audit or generation_tokens(selected) > final_limit:
        # Drop optional member detail before any member. All categories use
        # the same structure; no filename/entity/test-specific routing.
        if directory_compaction and not directory_names_attempted:
            from backend.sales.directory_evidence import compact_directory_packets
            selected = compact_directory_packets(selected, names_only=True)
            directory_names_attempted = True
            continue
        counts = {}
        for row in selected:
            name = str(row.get('document_name') or '')
            if name:
                counts[name] = counts.get(name, 0) + 1
        removable = None
        represented_counts = {}
        for row in selected:
            for key in row.get('requirement_ids', []):
                represented_counts[key] = represented_counts.get(key, 0)+1
        for row in reversed(selected):
            group = row.get('packing_group_id') or row['evidence_id']
            members = [r for r in selected
                       if (r.get('packing_group_id') or r['evidence_id']) == group]
            if any(represented_counts.get(key,0) <= sum(key in r.get('requirement_ids',[]) for r in members)
                   for r in members for key in r.get('requirement_ids', [])):
                continue
            if any(r['evidence_id'] in reserved or r.get('required_document_coverage')
                   or (question is None and (r.get('protected_goal_ids') or r.get('protected_relation_types')))
                   or 'conflict_group' in (r.get('protected_relation_types') or [])
                   or str(r.get('packing_group_id') or '').startswith(('conflict_','relation_'))
                   or (question is None and (r.get('source_type') == 'visual' or r.get('visual_description')))
                   or r.get('candidate_requires_pixel_verification') for r in members):
                continue
            by_document = {}
            for member in members:
                if question is not None and (member.get('source_type') == 'visual' or member.get('visual_description')) and member.get('document_id') not in required:
                    continue  # Gallery metadata is not a required document.
                if question is not None and member.get('document_id') not in required and not member.get('required_document_coverage'):
                    continue
                name = str(member.get('document_name') or '')
                if name:
                    by_document[name] = by_document.get(name, 0) + 1
            if any(counts[name] <= count for name, count in by_document.items()):
                continue
            if len(members) < len(selected):
                removable = {r['evidence_id'] for r in members}
                break
        if removable is None:
            # Cell reservations are a packing priority, NOT an unlimited lock.
            # Relax only these reservations if they alone prevent fitting;
            # conflict, source-integrity and mandatory file protections remain.
            if audit_tokens(selected) > audit_limit or generation_tokens(selected) > final_limit:
                for row in reversed(selected):
                    group = str(row.get('packing_group_id') or '')
                    atomic_group = group.startswith(('conflict_','relation_')) or (
                        bool(group) and sum(r.get('packing_group_id') == group for r in selected) > 1)
                    if (row['evidence_id'] not in reserved and row.get('requirement_ids')
                            and not atomic_group and 'conflict_group' not in (row.get('protected_relation_types') or [])
                            and not any(row.get(k) for k in (
                                'required_document_coverage','protected_goal_ids','candidate_requires_pixel_verification'))):
                        removed.append(row['evidence_id'])
                        selected = [r for r in selected if r['evidence_id'] != row['evidence_id']]
                        break
                else:
                    row = None
                if row is not None:
                    continue
                # Source diversity is a priority, not an unlimited lock on
                # one incidental representative per source type. Keep at
                # least one factual row; explicit files/collections remain
                # hard reservations. Remove whole rows, never fact fragments.
                for row in reversed(selected):
                    group = str(row.get('packing_group_id') or '')
                    atomic = group.startswith(('conflict_','relation_')) or (
                        bool(group) and sum(r.get('packing_group_id') == group for r in selected) > 1)
                    if (question is not None and row['evidence_id'] in reserved and row['evidence_id'] not in hard_reserved
                            and not atomic and not row.get('required_document_coverage')
                            and not any(str(r.get('packing_group_id') or '').startswith(('conflict_','relation_'))
                                        for r in selected)
                            and row.get('document_id') not in required
                            and not row.get('candidate_requires_pixel_verification')
                            and any(r is not row and r.get('facts_eligible') is not False
                                and r.get('evidence_scope') not in {'display_labels','document_index'}
                                and str(r.get('text') or '').strip() for r in selected)):
                        removed.append(row['evidence_id'])
                        selected = [r for r in selected if r['evidence_id'] != row['evidence_id']]
                        reserved.discard(row['evidence_id'])
                        break
                else:
                    row = None
                if row is not None:
                    continue
                # Conflict/relation groups are atomic, not unlimited locks
                # on incidental corpus sources. Drop a whole non-mandatory
                # group only as a last resort; never retain only one side.
                groups = {}
                for member in selected:
                    group = str(member.get('packing_group_id') or '')
                    if group.startswith(('conflict_','relation_')):
                        groups.setdefault(group,[]).append(member)
                for group,members in reversed(list(groups.items())):
                    if any(r['evidence_id'] in hard_reserved or r.get('required_document_coverage')
                            or r.get('document_id') in required or r.get('candidate_requires_pixel_verification')
                            or r.get('protected_goal_ids') for r in members):
                        continue
                    ids = {r['evidence_id'] for r in members}
                    if len(ids) < len(selected):
                        removed.extend(sorted(ids))
                        selected = [r for r in selected if r['evidence_id'] not in ids]
                        reserved.difference_update(ids)
                        row = members[0]
                        break
                if row is not None:
                    continue
            # Only overview snapshots may be reduced, and only before audit.
            # File identity stays; factual lookup/conflict groups stay intact.
            changed = False
            if compact_overview:
                for row in sorted(selected, key=lambda r:len(str(r.get('text') or '')), reverse=True):
                    if row['evidence_id'] in compacted or not row.get('overview_content') or row.get('protected_relation_types') or row.get('candidate_requires_pixel_verification'):
                        continue
                    replacement = compact_overview(row)
                    if replacement and replacement.get('text') != row.get('text'):
                        selected = [replacement if r['evidence_id'] == row['evidence_id'] else r for r in selected]
                        compacted.append(row['evidence_id']); changed = True; break
            if changed: continue
            # Only when full minimal lists plus mandatory content cannot fit:
            # retain whole names with truthful partial membership metadata.
            from backend.sales.directory_evidence import trim_directory_member
            for candidate in sorted(selected, key=lambda r:len(str(r.get('text') or '')), reverse=True):
                replacement=trim_directory_member(candidate)
                if replacement:
                    selected=[replacement if r is candidate else r for r in selected]
                    changed=True
                    break
            if changed: continue
            break
        removed.extend(sorted(removable))
        selected = [r for r in selected if r['evidence_id'] not in removable]
    audit = {'policy':'shared_prompt_budget_before_support_audit', 'max_prompt_tokens':ceiling,
             'audit_input_limit':audit_limit, 'final_input_limit':final_limit,
             'audit_prompt_tokens':audit_tokens(selected),
             'generation_reserved_prompt_tokens':generation_tokens(selected),
             'kept_evidence_ids':[r['evidence_id'] for r in selected],
             'removed_evidence_ids':removed, 'compacted_overview_ids':compacted}
    audit['fits_audit'] = audit['audit_prompt_tokens'] <= audit['audit_input_limit']
    audit['fits_final'] = audit['generation_reserved_prompt_tokens'] <= audit['final_input_limit']
    audit['budget_satisfied'] = audit['fits_audit'] and audit['fits_final']
    audit['compression'] = compression
    audit['compression_applied'] = not originally_fits
    audit['initial_budget_satisfied'] = originally_fits
    audit['budget_policy'] = 'check_complete_prompts_first_single_cpu_pack_if_overflow'
    audit['selection_strategy'] = ('target_and_aspect_alignment' if operation == 'comparison'
        else 'collection_file_section_coverage' if operation == 'summary' else 'bound_requirements')
    audit['collection_action'] = collection_action
    audit['directory_compaction'] = {'applied':directory_compaction,
        'policy':'membership_first', 'names_only_attempted':directory_names_attempted}
    audit['original_audit_prompt_tokens'] = original_audit_tokens
    # Performance target is observable, not permission to drop protected facts.
    audit['compression_target_tokens'] = min(target_tokens,audit_limit)
    audit['compression_target_satisfied'] = audit['audit_prompt_tokens'] <= audit['compression_target_tokens']
    audit['scope'] = scope
    audit['operation'] = operation
    audit['reserved_content_ids'] = sorted(reserved)
    covered = {row.get('document_id') for row in selected}
    audit['missing_target_document_ids'] = sorted(required-covered)
    audit['requirement_coverage'] = coverage(requirements, selected)
    from backend.sales.directory_evidence import directory_coverage
    audit['directory_coverage'] = directory_coverage(directory_original, selected)
    return selected, audit


def stage_support_candidates(question, candidates, *, goals=()):
    """Non-selecting handoff; the unified Context Engine packs exactly once."""
    rows = [{**row, 'text':compact_source_text(str(row.get('text') or ''))} for row in candidates]
    for row in rows:
        row['prepared_content_sha256'] = hashlib.sha256(row['text'].encode()).hexdigest()
    return rows, {'policy':'candidate_handoff_no_secondary_selection',
                  'candidate_count':len(rows),'selected_count':len(rows),
                  'kept_evidence_ids':[row['evidence_id'] for row in rows]}


def prepare_support(question, candidates, *, goals=()):
    rows = [{**row, 'text':compact_source_text(str(row.get('text') or ''))} for row in candidates]
    requirements = list(dict.fromkeys([question, *goals]))
    anchors = [terms(goal) for goal in requirements if terms(goal)]
    scores = [max((len(anchor & terms(row.get('text', ''))) / max(1, len(anchor))
                   for anchor in anchors), default=0) for row in rows]
    selected = set()
    required = {row.get('document_id') for row in rows if row.get('required_document_coverage')}
    for document_id in required:
        indexes = [i for i, row in enumerate(rows) if row.get('document_id') == document_id]
        # Summary/overview evidence was already selected by structure, not by
        # similarity to the word "summarise". Keep its coverage together.
        selected.update(indexes)
    semantic_mode = any(row.get('goal_support_scores') for row in rows)
    for requirement, anchor in zip([g for g in requirements if terms(g)], anchors):
        if rows:
            index = max(range(len(rows)), key=lambda i: (
                (rows[i].get('goal_support_scores') or {}).get(requirement, 0),
                len(anchor & terms(rows[i].get('text', '')))))
            support = (rows[index].get('goal_support_scores') or {}).get(requirement, 0)
            if (semantic_mode and support >= .5) or (not semantic_mode and
                    rows[index].get('source_type') != 'public_web' and anchor & terms(rows[index].get('text', ''))):
                selected.add(index)
    # Existing semantic scores remain primary when available. Missing lexical
    # overlap in cross-language evidence is never used as an unsupported verdict.
    ranked = sorted(range(len(rows)), key=lambda i: (
        -max((rows[i].get('goal_support_scores') or {}).values(), default=0),
        0 if rows[i].get('source_type') == 'public_web' else -scores[i],
        rows[i].get('web_original_rank', i) if rows[i].get('source_type') == 'public_web' else i))
    limit_chars = min(6500, 1400 + 800 * max(1, len(anchors)))
    if any(row.get('source_type') == 'public_web' for row in rows):
        limit_chars = max(limit_chars, 4000)
    # Without semantic scores, lexical overlap cannot prove a web candidate
    # irrelevant. Keep a bounded publisher-diverse pool for the support judge.
    # With scores, low-ranked sources receive no mandatory diversity quota.
    web = [i for i in ranked if rows[i].get('source_type') == 'public_web']
    diverse, deferred, publishers = [], [], set()
    for i in web:
        publisher = rows[i].get('publisher_group') or urlsplit(rows[i].get('source_url') or '').hostname or str(i)
        if publisher in publishers:
            deferred.append(i)
        else:
            publishers.add(publisher); diverse.append(i)
    if not semantic_mode:
        ranked = [i for i in ranked if i not in web] + diverse + deferred
    window_audit = {}
    for i in web:
        original_length = len(rows[i]['text'])
        # A conservative contiguous window retains neighbouring qualifiers;
        # protected conflicts/relations are never cut by this web-only bound.
        protected = str(rows[i].get('packing_group_id', '')).startswith(('conflict_', 'relation_'))
        if original_length > 800 and not protected and not rows[i].get('protected_relation_types'):
            from backend.sales.evidence_gate import question_excerpt
            rows[i]['text'] = question_excerpt(rows[i]['text'], question, limit=800)
            rows[i]['preview_truncated'] = True
        window_audit[i] = {'original_chars':original_length, 'selected_chars':len(rows[i]['text'])}
    best = max(scores, default=0)
    used = sum(len(str(rows[i].get('text', ''))) for i in selected)
    for i in ranked:
        row = rows[i]
        semantic = max((row.get('goal_support_scores') or {}).values(), default=0)
        relevant = (semantic >= .5 if semantic_mode else
                    row.get('source_type') == 'public_web' or not best or scores[i] >= best * .5)
        # Let the judge inspect the best available candidate when none scores
        # highly; this is an uncertain fallback, never a support verdict.
        relevant = relevant or (semantic_mode and not selected and i == ranked[0])
        if (i not in selected and relevant and
                (used + len(str(row.get('text', ''))) <= limit_chars or not selected)):
            selected.add(i)
            used += len(str(row.get('text', '')))
    # Preserve all members of a selected protected relation/conflict group.
    groups = {rows[i].get('packing_group_id') for i in selected
              if str(rows[i].get('packing_group_id', '')).startswith(('conflict_', 'relation_'))}
    for i, row in enumerate(rows):
        if row.get('packing_group_id') in groups:
            selected.add(i)
        if row.get('source_type') == 'visual' or row.get('visual_description'):
            selected.add(i)  # Labels are small; pixels have a separate budget.
    packed = [rows[i] for i in range(len(rows)) if i in selected]
    # Keep the canonical-source digest for provenance. The selected/cleaned
    # text is a different representation and gets its own immutable digest.
    for row in packed:
        row['prepared_content_sha256'] = hashlib.sha256(
            str(row.get('text') or '').encode('utf-8')).hexdigest()
    decisions = [{'evidence_id':row['evidence_id'], 'kept':i in selected,
                  'reason': ('protected_relation_group' if i in selected and row.get('packing_group_id') in groups else
                             'required_file_coverage' if i in selected and row.get('required_document_coverage') else
                             'semantic_candidate' if i in selected and semantic_mode else
                             'bounded_candidate_for_semantic_audit' if i in selected else
                             'unscored_candidate_not_support_verdict' if semantic_mode and not row.get('goal_support_scores') else
                             'low_semantic_score' if semantic_mode and max((row.get('goal_support_scores') or {}).values(), default=0)<.5 else
                             'context_budget_or_weak_lexical_hint'),
                  'semantic_score':max((row.get('goal_support_scores') or {}).values(), default=None),
                  'lexical_hint':round(scores[i], 3), **window_audit.get(i, {})}
                 for i,row in enumerate(rows)]
    from backend.sales.stage_details import begin, end
    handle = begin('support_selection', {'question':question,
        'candidate_ids':[row['evidence_id'] for row in rows]})
    end(handle, {'decisions':decisions, 'semantic_ranking_available':semantic_mode,
                 'kept_evidence_ids':[row['evidence_id'] for row in packed]})
    return packed, {'policy': 'question_sized_protected_support_v2',
                    'decisions':decisions, 'semantic_ranking_available':semantic_mode,
                    'candidate_count': len(rows), 'selected_count': len(packed),
                    'selected_chars': sum(len(str(row.get('text', ''))) for row in packed),
                    'target_chars': limit_chars, 'score_is_support_verdict': False,
                    'kept_evidence_ids': [row['evidence_id'] for row in packed],
                    'removed_evidence_ids': [row['evidence_id'] for i, row in enumerate(rows) if i not in selected]}


def support_signature(question, rows, goals):
    def stable(value):
        if isinstance(value, dict):
            return {key:stable(item) for key,item in value.items()
                    if key not in {'elapsed_ms', 'input_tokens', 'reused'}}
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        return value
    return hashlib.sha256(json.dumps([question, goals, stable(rows)], sort_keys=True,
                         ensure_ascii=False, default=str).encode()).hexdigest()


def audit_rows(rows, evidence_by_id):
    return [{**row, 'citations': evidence_by_id.get(row['evidence_id'], {}).get('citations', [])}
            for row in rows]


def gallery_label_evidence(assets):
    """Display-only evidence: identities and labels, never engineering facts."""
    captions, citations = [], []
    for asset in assets:
        labels = asset.get('image_labels') or {}
        caption = labels.get('variant_or_code') or labels.get('product_name') or asset.get('customer_title')
        if caption:
            captions.append(str(caption))
        if asset.get('citation'):
            citations.append(asset['citation'])
    return {'evidence_id':'G1', 'text':'知识库展示图片标签：'+'；'.join(dict.fromkeys(captions)),
            'source_type':'catalog_labels', 'evidence_scope':'display_labels',
            'gallery_collection':True, 'facts_eligible':False,
            'claim_scope':'asset_identity_only_not_performance', 'citations':citations}
