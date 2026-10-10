"""Customer-only visual policy. Asset metadata is navigation, not pixel evidence."""
import hashlib
import json


def asset_key(asset):
    return 'customer:' + str(asset.get('document_id') or '') + ':' + str(asset.get('visual_id') or '')


def balanced_assets(assets, limit=4):
    unique = {}
    for asset in assets:
        if asset.get('asset_id') or not asset.get('document_id') or not asset.get('visual_id'):
            continue
        unique.setdefault(asset_key(asset), asset)
    buckets = {}
    for asset in unique.values():
        buckets.setdefault(asset['document_id'], []).append(asset)
    result = []
    while any(buckets.values()) and len(result) < limit:
        for bucket in buckets.values():
            if bucket and len(result) < limit:
                result.append(bucket.pop(0))
    return result


def wants_inspection(plan):
    return bool('customer_documents' in plan.tools and
        (plan.document_visual_required is True or plan.recovery_directive.get('customer_visual_recovery')))


def recovery_available(assets, states):
    return sum(s.get('pixel_calls',0) for s in states.values()) < 4 and any(
        isinstance(a.get('image_bytes'),bytes) and bool(a['image_bytes']) and
        states.get(asset_key(a),{}).get('status','unrecognized') == 'unrecognized'
        for a in balanced_assets(assets,24))


def visual_manifest(assets, states=None):
    states = states or {}
    return [dict(asset_id=asset_key(a), document_id=a['document_id'], visual_id=a['visual_id'],
        document_name=a.get('document_name'), source=a.get('source') or {},
        nearby_text=str(a.get('searchable_text') or a.get('explanation') or '')[:350],
        **states.get(asset_key(a), {'status':'unrecognized'})) for a in balanced_assets(assets, 24)]


def cache_key(content, question):
    return hashlib.sha256(content).hexdigest() + ':' + hashlib.sha256(question.encode()).hexdigest()


def display_evidence(assets):
    """One source-bound display inventory per file; never supports image details."""
    groups = {}
    for a in balanced_assets(assets):
        groups.setdefault(a['document_id'], []).append(a)
    rows = []
    for document, group in groups.items():
        identity = 'VG' + hashlib.sha256(document.encode()).hexdigest()[:12]
        rows.append({'evidence_id':identity,'document_id':document,'document_name':group[0].get('document_name'),
            'source_type':'customer_document','evidence_scope':'display_labels','gallery_collection':True,
            'facts_eligible':False,'claim_scope':'image_presence_and_source_location_only_not_image_content',
            'original_chunk_id':'display_inventory:'+document,
            'source_refs':[a['visual_id'] for a in group],
            'text':json.dumps({'display_images':[{'visual_id':a['visual_id'], 'source':a.get('source') or {}}
                                              for a in group]},ensure_ascii=False),
            'citations':[{'document_name':group[0].get('document_name'),'source_type':'customer_upload_visual'}]})
    return rows


def delivery_envelope(rows, assets, states, *, display_requested=True):
    return {'visual_evidence':[r['evidence_id'] for r in rows if r.get('description_basis')=='pixel_tool_observation'],
            'display_images':[{'asset_id':asset_key(a),'document_id':a['document_id'],'visual_id':a['visual_id']}
                              for a in balanced_assets(assets)] if display_requested else [],
            'visual_status':visual_manifest(assets,states),
            'policy':'customer_only_inspect_on_request_or_one_shared_recovery; audit_and_final_text_only'}
