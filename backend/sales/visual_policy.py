"""Question-time visual support, not format/topic routing or an extra model call."""
from pathlib import Path
from backend.sales.gallery_ranking import COMPANY_GALLERY_DISPLAY_LIMIT


def company_metadata_audit(assets):
    """Cheap catalogue checks, not image interpretation or semantic approval.

    ACL and file decoding are enforced separately at materialization. Missing
    descriptive labels are disclosed, not guessed or silently called reviewed.
    """
    import time
    started = time.perf_counter()
    seen, duplicates, incomplete, disputed, ready = set(), [], [], [], []
    for asset in assets:
        key = str(asset.get('asset_id') or '')
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        citation = asset.get('citation') or {}
        if not key or not asset.get('customer_title') or not citation.get('document_name'):
            incomplete.append(key)
        labels = asset.get('image_labels') or {}
        if labels.get('label_issues'):
            disputed.append(key)
        if labels.get('display_ready') and not labels.get('label_issues'):
            ready.append(key)
    return {'policy':'company_visual_lightweight', 'method':'cpu_metadata_checks',
            'candidate_count':len(assets), 'duplicate_asset_ids':duplicates,
            'incomplete_label_asset_ids':incomplete, 'model_calls':0,
            'disputed_label_asset_ids':disputed, 'metadata_display_ready_asset_ids':ready,
            'semantic_verified':False, 'pixel_contents_verified':False,
            'acl_and_file_checks':'enforced_before_pixel_input',
            'elapsed_ms':round((time.perf_counter()-started)*1000, 3)}


def asset_key(asset):
    if asset.get('asset_id'):
        return 'company:' + str(asset['asset_id'])
    return 'customer:' + str(asset.get('document_id', '')) + ':' + str(asset.get('visual_id', ''))


def labelled_company_gallery(plan, sources, *, has_direct_image=False, limit=COMPANY_GALLERY_DISPLAY_LIMIT):
    """A narrow display contract, not a bypass for technical or external tasks.

    Sources are already retrieved under ACL. Every selected asset must also
    have source-derived labels; missing/disputed labels fall back to normal audit.
    """
    if (has_direct_image or set(sources) != {'company_rag'} or set(plan.tools) != {'company_rag'}
            or not plan.wants_visuals or plan.visual_task != 'gallery' or plan.product_overview
            or plan.task_type not in {'unknown', 'factual_lookup', 'case_reference'}
            or plan.intent not in {'unknown', 'case_reference', 'product_parameter'}):
        return None
    assets = list((sources.get('company_rag') or {}).get('visual_assets') or [])
    if plan.visual_scope == 'product':
        if plan.gallery_scope not in {'category', 'exact'}:
            return None
        if plan.gallery_scope == 'exact' and not plan.gallery_target.strip():
            return None
        if plan.gallery_product_family == 'unknown' and plan.gallery_scope != 'exact':
            return None
        if plan.gallery_product_family in {'decorative_panel','insulated_panel'}:
            assets = [a for a in assets if (a.get('image_labels') or {}).get('product_family_key')
                      == plan.gallery_product_family]
        if plan.gallery_scope == 'exact':
            target=''.join(plan.gallery_target.casefold().split())
            assets=[a for a in assets if any(target in ''.join(str((a.get('image_labels') or {}).get(field) or '').casefold().split())
                      for field in ('variant_or_code','product_name'))]
    if not assets:
        return None
    audit = company_metadata_audit(assets)
    if (audit['incomplete_label_asset_ids'] or audit['disputed_label_asset_ids']
            or len(audit['metadata_display_ready_asset_ids']) != len(assets)):
        return None
    # Keep retrieval order within each variant, but offer diverse labels before duplicates.
    unique, repeated, seen, hashes = [], [], set(), set()
    for asset in assets:
        labels = asset.get('image_labels') or {}
        if labels.get('label_basis') != 'existing_source_metadata_and_explicit_catalogue_joins':
            return None
        if (not labels.get('citation') or not labels.get('asset_id') == asset.get('asset_id')
                or asset.get('origin') not in (None, 'company_rag')):
            return None
        digest = labels.get('sha256') or asset['asset_id']
        if digest in hashes:
            continue
        hashes.add(digest)
        group = (labels.get('category'), labels.get('variant_or_code') or labels.get('product_name'),
                 tuple(labels.get('project_names') or []))
        (repeated if group in seen else unique).append(asset)
        seen.add(group)
    selected = (unique + repeated)[:limit]
    if not selected:
        return None
    captions, citations = [], []
    for index, asset in enumerate(selected, 1):
        labels = asset['image_labels']
        caption = (labels.get('variant_or_code') or '、'.join(labels.get('project_names') or [])
                   or labels.get('product_name') or '资料图片')
        captions.append(f'{index}. {caption}')
        citations.append({**asset['citation'], 'evidence_id':f'V{index}', 'asset_id':asset['asset_id']})
    return {'intent':'case_reference' if plan.visual_scope == 'case' else 'unknown',
            'answerable':True, 'customer_reply':'以下是按知识库标签找到的图片：\n'+'\n'.join(captions)
                +'\n本次展示召回到的图片，不代表完整产品目录；标签不作为性能或合规证明。',
            'normalized_terms':[], 'key_points':[], 'citations':citations, 'missing_information':[],
            'risk_warnings':[], 'next_action':'', 'image_observations':[], 'online_sources':[],
            'visual_assets':selected, 'retrieval':{'result_count':len(selected),
                'supporting_results':[], 'visual_count':len(selected)},
            'meta':{'model_used':False,'mode':'company_gallery_labels',
                'evidence_gate':{**audit,'status':'supported','visual_need':'not_required',
                    'evaluation_method':'labelled_company_gallery','semantic_verified':False,
                    'pixel_contents_verified':False}}}


def visual_candidates(sources, limit=8):
    buckets = []
    for source in sources.values():
        if not isinstance(source, dict):
            continue
        candidates = source.get('selected_visuals') or source.get('visual_assets') or []
        buckets.append(list(candidates))
    result, seen = [], set()
    while any(buckets) and len(result) < limit:
        for bucket in buckets:
            if not bucket:
                continue
            asset = bucket.pop(0)
            key = asset_key(asset)
            if key in seen:
                continue
            seen.add(key)
            result.append({'asset_id':key, 'title':str(asset.get('customer_title') or asset.get('document_name') or '')[:160],
                           'context':str(asset.get('searchable_text') or asset.get('explanation') or '')[:450],
                           'location':asset.get('source') or asset.get('citation') or {},
                           'image_labels':{name:(asset.get('image_labels') or {}).get(name) for name in
                                           ('category','product_name','variant_or_code','project_names','cities',
                                            'display_ready','label_issues','label_basis','pixel_contents_verified')},
                           'pixels_supplied':False})
            if len(result) >= limit:
                break
    return result


def validate_visual_audit(raw, candidates):
    need = raw.get('visual_need', 'uncertain')
    if need not in {'required', 'not_required', 'uncertain'}:
        raise ValueError('invalid_visual_audit_need')
    allowed = {c['asset_id'] for c in candidates}
    purpose = raw.get('visual_purpose','uncertain')
    if purpose not in {'display_only','content_inspection','label_dispute','unreadable','uncertain'}:
        raise ValueError('invalid_visual_audit_purpose')
    ids = raw.get('candidate_asset_ids', [])
    if not isinstance(ids, list):
        raise ValueError('invalid_visual_audit_ids')
    if any(not isinstance(i, str) or i not in allowed for i in ids):
        raise ValueError('invalid_visual_audit_reference')
    valid = [i for i in ids if isinstance(i, str) and i in allowed] if isinstance(ids, list) else []
    return {'visual_need':need, 'candidate_asset_ids':list(dict.fromkeys(valid)),
            'visual_purpose':purpose,
            'invalid_visual_ids': [str(i)[:100] for i in ids if not isinstance(i, str) or i not in allowed] if isinstance(ids, list) else [],
            'visual_reason':str(raw.get('visual_reason') or '')[:300]}


def enforce_company_label_policy(raw, *, company_only=False):
    """Backend source policy: company pixels are not a recoverable input.

    Preserve the semantic verdict/missing aspects; this is not certification of
    tags or permission to fabricate facts. Customer visual gaps remain intact.
    """
    result = dict(raw)
    if not company_only or result.get('error_code'):
        return result
    result.update(visual_need='not_required',visual_purpose='display_only',
                  candidate_asset_ids=[],pixel_support_pending=False,
                  visual_input_policy='company_labels_only')
    if result.get('recovery_need') == 'visual':
        result['recovery_need']='none'
    if result.get('gap_type') == 'visual_input':
        result['gap_type']='local_evidence'
    return result


def enforce_requested_visual_support(raw, candidates, *, required=False, descriptions_ready=False):
    """An explicit task requirement cannot disappear behind unrelated text support.

    No keyword or file-format routing. False/unknown never vetoes the auditor.
    Available IDs still refer only to metadata; pixel support is pending.
    """
    result = dict(raw)
    if descriptions_ready:
        # The semantic auditor has already read source-bound visual evidence.
        # Its genuine gaps remain intact; not seeing pixels is not a gap.
        result['pixel_support_pending'] = False
        result['visual_input_policy'] = 'textual_evidence_only'
        return result
    if not required or result.get('error_code'):
        return result
    checked = validate_visual_audit(result, candidates)
    customer_ids = [c['asset_id'] for c in candidates
                    if c['asset_id'].startswith('customer:')]
    selected = list(dict.fromkeys([*checked['candidate_asset_ids'], *customer_ids])) if customer_ids else []
    result.update(visual_need='required', visual_purpose='content_inspection',
                  candidate_asset_ids=selected, pixel_support_pending=False,
                  recovery_need='visual', visual_input_policy='textual_evidence_only',
                  missing_aspects=result.get('missing_aspects') or ['所需视觉内容尚未生成带来源的描述'])
    if result.get('status') in {'supported', 'partial'}:
        result['status'] = 'partial' if selected and result.get('supported_evidence_ids') else 'unsupported'
    if not selected:
        result.update(gap_type='visual_input',
                      problem='Required visual input unavailable',
                      missing_aspects=['Required attachment visual content is unavailable'])
    return result


def select_visual_support(candidates, audit, *, limit=4, mandatory_keys=()):
    """Select external pixels only; company assets are label/display inputs.

    Customer OCR checks survive planner hints. Company policy is backend-owned
    and cannot be overridden by a planner/auditor's request for inspection.
    """
    need = audit.get('visual_need', 'uncertain')
    purpose = audit.get('visual_purpose','uncertain')
    preferred = set(audit.get('candidate_asset_ids') or [])
    mandatory = set(mandatory_keys)
    disputed = {asset_key(a) for a in candidates if a.get('asset_id') and
                (a.get('image_labels') or {}).get('label_issues')}
    # Healthy company gallery labels may be displayed without pixel inference.
    # Required diagram/chart/detail tasks still override metadata-only display.
    display_only = {asset_key(a) for a in candidates if a.get('asset_id') and
                    (a.get('image_labels') or {}).get('display_ready') and
                    not (a.get('image_labels') or {}).get('label_issues')}
    eligible = [a for a in candidates if not a.get('asset_id')
                and (need != 'not_required' or asset_key(a) in mandatory or asset_key(a) in disputed)
                and (not preferred or asset_key(a) in preferred or asset_key(a) in mandatory)]
    eligible.sort(key=lambda a: (asset_key(a) not in mandatory, asset_key(a) not in preferred))
    # Preserve mandatory pixel checks first, then share remaining slots between
    # sources. File arrival order must not consume the entire visual budget.
    ordered = [a for a in eligible if asset_key(a) in mandatory]
    buckets = {}
    for asset in eligible:
        if asset_key(asset) not in mandatory:
            origin = 'company' if asset.get('asset_id') else 'customer'
            buckets.setdefault(origin, []).append(asset)
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:
                ordered.append(bucket.pop(0))
    selected, seen = [], set()
    for asset in ordered:
        key = asset_key(asset)
        if key not in seen and len(selected) < limit:
            selected.append(asset)
            seen.add(key)
    return selected, {'visual_need':need, 'selected_asset_ids':[asset_key(a) for a in selected],
                      'budget_excluded_asset_ids':[asset_key(a) for a in eligible if asset_key(a) not in seen],
                      'required_but_missing':need == 'required' and not selected and purpose != 'display_only'
                          and not audit.get('visual_input_policy') == 'company_labels_only'
                          and not (candidates and all(a.get('asset_id') for a in candidates)),
                      'selection_method':'existing_evidence_audit_plus_pixel_verification',
                      'metadata_only_display_asset_ids':sorted(display_only - mandatory),
                      'disputed_label_asset_ids':sorted(disputed),
                      'visual_purpose':purpose,
                      'company_pixel_permission':'never_labels_only',
                      'planner_hint_is_veto':False}


def materialize_company_assets(assets, resolve_asset, max_bytes=15*1024*1024):
    """Read only ACL-filtered indexed assets, never a model-provided path/URL."""
    from PIL import Image
    output, failures = [], []
    for asset in assets:
        key = str(asset.get('asset_id') or '')
        indexed = resolve_asset(key)
        if not indexed:
            failures.append({'asset_id':key,'error':'not_indexed_or_not_authorised'})
            continue
        try:
            path = Path(indexed.get('image_path') or '')
            if not path.is_file() or path.stat().st_size > max_bytes:
                raise ValueError('missing_or_oversized_image')
            content = path.read_bytes()
            from io import BytesIO
            with Image.open(BytesIO(content)) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError('image_pixel_limit')
                media_type = Image.MIME.get(image.format, 'image/png')
                image.verify()
            citation = dict(asset.get('citation') or indexed.get('citation') or {})
            output.append({**asset, 'image_bytes':content, 'media_type':media_type,
                           'visual_id':key, 'document_name':citation.get('document_name') or asset.get('customer_title') or key,
                           'source':citation, 'kind':'company_image', 'origin':'company_rag'})
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            failures.append({'asset_id':key,'error':type(exc).__name__})
    return output, failures


def input_manifest(paths, visuals, direct=None):
    """Actual input order, with all matching evidence IDs after pixel deduplication."""
    import hashlib
    rows = []
    for index, path in enumerate(paths, 1):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = [v for v in visuals if v.get('image_bytes') and
                   hashlib.sha256(v['image_bytes']).hexdigest() == digest]
        is_direct = direct is not None and hashlib.sha256(direct.read_bytes()).hexdigest() == digest
        rows.append({'image_id':f'image{index}', 'image_number':index, 'origin':'direct_upload' if is_direct else 'retrieved_asset',
                     'source_candidates':[{'evidence_id':v.get('evidence_id'), 'visual_id':v.get('visual_id'),
                                           'asset_id':v.get('asset_id'), 'origin':v.get('origin','customer_documents'),
                                           'document_name':v.get('document_name'), 'source':v.get('source')}
                                          for v in matches], 'sha256':digest})
    return rows


def unavailable_visual_response(audit):
    """No snippet fallback may claim to have inspected unavailable pixels."""
    supplied = bool(audit.get('actual_pixel_count'))
    return {'intent':'unknown', 'answerable':False,
            'customer_reply':('本轮已输入原图，但生成回答未通过证据或引用校验，暂不能给出可靠结论。' if supplied else
                              '已找到相关图片线索，但所需图片未能进入本轮视觉输入，暂不能可靠分析图片内容。'),
            'normalized_terms':[], 'key_points':[], 'citations':[],
            'missing_information':['通过证据与引用校验的视觉回答' if supplied else '所需原图的实际视觉输入'],
            'risk_warnings':[], 'next_action':'请重试或重新上传所需原图。',
            'image_observations':[], 'visual_assets':[],
            'retrieval':{'result_count':0, 'supporting_results':[], 'visual_count':0},
            'meta':{'model_used':supplied, 'mode':'visual_answer_validation_failed' if supplied else 'required_visual_input_unavailable',
                    'visual_support':audit,
                    'error':{'code':'GROUNDING_FAILED' if supplied else 'MISSING_VISUAL_INPUT',
                             'stage':'validate_answer' if supplied else 'visual_input', 'retryable':not supplied}}}


def generation_evaluation(audit, image_count):
    """Project pre-pixel diagnostics into current input state, not a new verdict.

    Preserve the original audit outside the prompt for recovery and inspection.
    A statement that pixels were missing BEFORE generation must not be presented
    as a current fact once those pixels have arrived.
    """
    if not image_count or not audit.get('pixel_support_pending'):
        return {key:audit[key] for key in ('status','supported_evidence_ids','conflict_groups','missing_aspects','problem','reason','support_spans',
            'visual_need','visual_purpose','source_constraint','pixel_support_pending','actual_pixel_count') if key in audit}
    return {'status':'uncertain', 'evaluation_phase':'pre_pixel_metadata_only',
            'visual_need':audit.get('visual_need'), 'missing_aspects':audit.get('missing_aspects', []),
            'pixel_support_pending':True, 'actual_pixel_count':image_count, 'semantic_verified':False,
            'problem':'Visual support awaits inspection of the NOW attached pixels.',
            'supported_evidence_ids':[], 'previous_audit_retained_server_side':True}
