"""Backend delivery state; separate task intent from evidence and pixels."""

IMAGE_DELIVERY_POLICY = '''image_delivery为本轮后端配图状态，不证明性能。requested=true且available_count>0时可结合delivered标题/明确关联分析，说明图片见下方；不重复卡片或输出链接/Markdown/HTML图片/内部ID/路径，实际展示由后端处理，没像素或内嵌图不代表无图片。公司图只用标签。
requested=true且available_count=0说明本轮配图缺口，保留已支持文字；false不继承旧配图。company_relations仅已读SQL导航，不证明性能，未列出不表示数据库无关系。'''


def image_delivery(plan, assets):
    from backend.sales.review_execution import display_associations
    delivered = [{'evidence_id': a.get('evidence_id'),
                  'asset_id': a.get('asset_id') or a.get('visual_id'),
                  'title': a.get('title') or a.get('document_name') or '',
                  'associations': display_associations(a)}
                 for a in assets]
    return {'requested': bool(plan and plan.wants_visuals),
            'available_count': len(delivered), 'delivered': delivered,
            'company_policy': 'labels_only_no_runtime_pixels',
            'state_basis': 'server_selected_display_assets_not_pixel_input'}


def relation_delivery(retrieval):
    scope = (retrieval.get('meta') or {}).get('relation_scope') or {}
    relations = scope.get('relations') or []
    fields = ('relation', 'product_id', 'case_id', 'entity_id', 'family',
              'section_id', 'document_key', 'asset_id', 'reference_code')
    return {'basis': 'sql_navigation_not_fact_proof', 'count': len(relations),
            'types': sorted({r['relation'] for r in relations if r.get('relation')}),
            'links': [{k: r[k] for k in fields if k in r} for r in relations[:12]],
            'omitted_link_count': max(0, len(relations)-12)}
