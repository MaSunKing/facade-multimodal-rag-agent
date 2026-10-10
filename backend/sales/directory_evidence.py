"""CPU directory membership packets; never infer properties from relations."""
import hashlib
import json
from collections import defaultdict
from backend.sales.content_taxonomy import canonical_domains, LABELS


def collection_packets(category, members, *, packet_chars=2400):
    """Whole members only. Canonical IDs stay server-side, coverage survives packing."""
    members = list({m['id']: m for m in members}.values())
    if not members:
        return []
    expected = [m['id'] for m in members]
    batches = [[]]
    size = 0
    for member in members:
        length = len(member['text'])
        if batches[-1] and size+length > packet_chars:
            batches.append([])
            size = 0
        batches[-1].append(member)
        size += length
    output = []
    for batch in batches:
        ids = [m['id'] for m in batch]
        refs = [r for m in batch for r in m.get('citations', [])]
        output.append({'evidence_id':'D'+hashlib.sha256(repr((category, ids)).encode()).hexdigest()[:16],
            'source_type':'catalog_sql','catalog_record_type':'directory_collection',
            'directory_collection':True,'directory_category':category,
            'directory_member_ids':ids,'directory_expected_ids':expected,
            'directory_members':[dict(m) for m in batch],
            'directory_item_count':len(ids),'facts_eligible':True,
            'entity_coverage_ids':ids if category in {'02_product_information','05_project_case'} else [],
            'fact_support_scope':'registered_directory_membership_only_not_performance_or_full_content',
            'text':f'本轮{LABELS[category]}目录集合，共{len(expected)}项；本包{len(ids)}项。仅为已登记名称与属性，不推断未登记组合或全文内容。\n'+
                '\n'.join(m['text'] for m in batch),
            'citations':list({json.dumps(r,ensure_ascii=False,sort_keys=True):r for r in refs}.values())})
    return output


def process_collection(category, members, request, *, packet_chars=2400):
    """Aggregate a complete resolved public set before token packing.

    Membership IDs stay attached even to aggregate packets. Missing attributes
    form an explicit unknown bucket; they never silently reduce the denominator.
    """
    action = request.get('action', 'none')
    fields = request.get('group_by', [])
    members = list({m['id']:m for m in members}.values())
    if action == 'none' or not fields and action == 'list':
        return collection_packets(category, members, packet_chars=packet_chars)
    groups = defaultdict(list)
    for member in members:
        values = member.get('attributes', {})
        group = tuple(str(values.get(f) or '').strip() or '未登记' for f in fields)
        groups[group].append(member)
    # No rows is no reliable scoped set, not proof that this company has zero projects.
    if not members:
        return []
    expected = [m['id'] for m in members]
    if action == 'count':
        # Pack many small group counts together, not one metadata-heavy evidence
        # row per group. Coverage still maps to actual members, not group IDs.
        entries=[]
        for group, rows in sorted(groups.items()):
            key='group_'+hashlib.sha256(repr(group).encode()).hexdigest()[:16]
            entries.append({'id':key,'text':('；'.join(f'{f}={v}' for f,v in zip(fields,group)) or '全部')+f'：{len(rows)}项',
                'members':[m['id'] for m in rows],
                'group':{'values':dict(zip(fields,group)),'count':len(rows)},
                'citations':[r for m in rows for r in m.get('citations',[])]})
        lookup={e['id']:e for e in entries}
        packets=collection_packets(category,entries,packet_chars=packet_chars)
        for p in packets:
            batch=[lookup[i] for i in p['directory_member_ids']]
            ids=[i for e in batch for i in e['members']]
            p.update(directory_member_ids=ids,directory_expected_ids=expected,
                directory_item_count=len(ids),entity_coverage_ids=ids if category in {'02_product_information','05_project_case'} else [],
                catalog_record_type='directory_aggregate',collection_action=action,
                collection_group_by=fields,collection_total_count=len(expected),collection_group_count=len(ids),
                collection_groups=[e['group'] for e in batch],
                collection_missing_fields=sorted({f for e in batch for f,v in e['group']['values'].items() if v=='未登记'}))
            p['text']=f'已登记目录统计，总数={len(expected)}，按目录成员ID去重；空属性归入未登记，不代表企业全部实际对象。\n'+'\n'.join(e['text'] for e in batch)
            p['evidence_id']='D'+hashlib.sha256(repr((category,action,fields,ids)).encode()).hexdigest()[:16]
        return packets
    output = []
    for group, rows in sorted(groups.items()):
        label = '；'.join(f'{f}={v}' for f,v in zip(fields,group)) or '全部'
        ids = [m['id'] for m in rows]
        text = f'已登记目录统计：{label}；去重成员数={len(ids)}；本轮目录集合总数={len(expected)}。统计单位为目录成员ID，不是图片、章节出现次数或企业全部实际对象；空属性归入未登记。'
        if action == 'list':
            # Preserve whole-member packet boundaries for potentially huge grouped lists.
            packets = collection_packets(category, rows, packet_chars=packet_chars)
            for p in packets:
                p['text'] = label+'\n'+p['text']
        for p in packets:
            p.update(directory_expected_ids=expected, catalog_record_type='directory_aggregate' if action=='count' else 'directory_collection',
                collection_action=action, collection_group_by=fields,
                collection_total_count=len(expected), collection_group_count=len(ids),
                collection_missing_fields=[f for f,v in zip(fields,group) if v=='未登记'])
            p['evidence_id'] = 'D'+hashlib.sha256(repr((category,action,fields,group,p['directory_member_ids'])).encode()).hexdigest()[:16]
        output.extend(packets)
    return output


def from_directory(connection, scope, categories, collection=None, *, question='', operation='summary', scorer=None):
    """Consume backend-resolved IDs, never model-proposed IDs or unrestricted fallback."""
    from backend.sales.catalog_store import _product_citations, _case_citations
    domains = canonical_domains(categories)
    # Smaller atomic packets for multi-category requests let exact packing
    # retain one representative per category without locking six large packets.
    # This is a packet-size heuristic, not a replacement for token preflight.
    packet_chars = max(300, 2400 // max(1, len(domains)))
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ids = set(scope.get('entity_ids', [])) | set(scope.get('directory_member_ids', []))
    if not ids and operation=='lookup' and '02_product_information' in domains and 'products' in tables:
        ids={r[0] for r in connection.execute("SELECT product_id FROM products WHERE access_scope='public'")}
    output = []
    from backend.sales.catalog_structure import read_product_attributes
    attributes={r['product_id']:r for r in read_product_attributes(connection, sorted(ids))} if 'products' in tables else {}
    for table, key, name, category, columns, citations in (
        ('products','product_id','name','02_product_information',
         [('family','系列'),('surface_finish','饰面'),('insulation_type','保温层')],_product_citations),
        ('project_cases','case_id','project_name','05_project_case',
         [('province','省份'),('city','城市'),('project_category','项目类型'),
          ('product_name','登记产品'),('installation_method','登记施工方法')],_case_citations)):
        if table not in tables or (domains and category not in domains):
            continue
        members = []
        for row in connection.execute(f"SELECT * FROM {table} WHERE access_scope='public' ORDER BY {name}"):
            if row[key] not in ids:
                continue
            text = row[name] + ''.join('；'+label+'：'+str(row[column]) for column,label in columns if row[column])
            if table=='products':
                attr=attributes.get(row[key], {})
                from backend.sales.finish_options import finish_option_text
                options = finish_option_text(connection, row[key])
                if options:
                    text += '；' + options
                from backend.sales.product_source_profiles import product_profile_text
                profile_text = product_profile_text(connection, row[key], include_parameters=False)
                if profile_text:
                    text += '；' + profile_text
                text += ''.join('；'+label+'：'+str(attr[column]) for column,label in
                    [('color_name','花色'),('product_form','产品形态')] if attr.get(column))
            attrs = {column:row[column] for column,label in columns}
            attrs.update(category=category)
            if table=='products':
                attrs.update(attributes.get(row[key], {}))
            refs = citations(connection,row)
            if table=='products':
                from backend.sales.product_source_profiles import product_profile_citations
                refs = [*refs, *product_profile_citations(connection,row[key])]
            members.append({'id':row[key],'name':row[name],'text':text,'attributes':attrs,
                'attribute_labels':dict(columns),'citations':refs})
        if table=='products' and question and (collection or {}).get('action','none')=='none':
            from backend.sales.directory_semantics import locate_view, hierarchy_members
            from backend.sales.product_source_profiles import read_product_profile
            selection=locate_view(question,scorer=scorer)
            profiles={m['id']:read_product_profile(connection,m['id']) for m in members}
            parents=hierarchy_members(members,selection,profiles)
            packets=collection_packets(category,parents,packet_chars=packet_chars)
            for p in packets:
                p['entity_coverage_ids']=[i for m in p['directory_members'] for i in m['child_ids']]
                p['directory_level']='family_with_requested_options'
                p['directory_semantic_selection']=selection
                p['text']='按产品体系组织，花色／饰面／芯材是属性，不按底层条目数量统计产品种类。\n'+p['text']
            output.extend(packets)
        else:
            output.extend(process_collection(category,members,collection or {},packet_chars=packet_chars))
    # Technical/enterprise directories are section/document objects, not fake product IDs.
    topics = set(domains)-{'02_product_information','05_project_case'}
    candidates = set(scope.get('candidate_ids', []))
    # Retrieval omits the large candidate-ID list from public metadata. Its
    # trusted SQL category-scope marker still authorizes a public section scan;
    # an empty/unresolved scope alone never authorizes all-library fallback.
    topic_scope = scope.get('topic_directory') or {}
    collection_topics = (set(canonical_domains(topic_scope.get('categories', []))) & topics
        if topic_scope.get('basis') == 'section_evidence_category' else set())
    required_tables = {'knowledge_text_evidence','knowledge_evidence_sources','knowledge_documents'}
    if not topics or not (candidates or collection_topics) or not required_tables <= tables:
        return output
    grouped = defaultdict(dict)
    sql = '''SELECT e.evidence_id,e.metadata_json,s.document_key,s.section_heading,
                    s.reference_json,d.document_name
             FROM knowledge_text_evidence e JOIN knowledge_evidence_sources s USING(evidence_id)
             JOIN knowledge_documents d USING(document_key) WHERE e.access_scope='public'
             ORDER BY d.document_name,s.section_heading,e.evidence_id'''
    for row in connection.execute(sql):
        domains_here = set(canonical_domains(json.loads(row['metadata_json'] or '{}').get('knowledge_domains', [])))
        if row['evidence_id'] not in candidates and not (collection_topics & domains_here):
            continue
        heading = str(row['section_heading'] or row['document_name'])
        for category in sorted(topics & domains_here):
            identity = category+'\0'+row['document_key']+'\0'+heading
            key = 'section_'+hashlib.sha256(identity.encode()).hexdigest()[:16]
            member = grouped[category].setdefault(key,{'id':key,'text':row['document_name']+'／'+heading,'citations':[],
                'attributes':{'category':category,'document_name':row['document_name'],'section_heading':heading}})
            member['citations'].append(json.loads(row['reference_json'] or '{}'))
    if {'visual_assets','visual_directory'} <= tables:
        for row in connection.execute('''SELECT v.asset_id,v.document_name,v.source_page,d.category,d.text_label
            FROM visual_assets v JOIN visual_directory d USING(asset_id) WHERE v.access_scope='public'
            ORDER BY d.category,v.asset_id'''):
            category={'node':'04_node_atlas','process':'03_construction_method'}.get(row['category'])
            if category not in topics or 'visual:'+row['asset_id'] not in candidates:
                continue
            key='visual:'+row['asset_id']
            grouped[category][key]={'id':key,'text':'图片目录标签：'+row['text_label'],
                'attributes':{'category':category,'document_name':row['document_name']},
                'citations':[{'document_name':row['document_name'],'source_page':row['source_page']}]}
    for category, members in grouped.items():
        output.extend(process_collection(category,list(members.values()),collection or {},packet_chars=packet_chars))
    return output


def directory_coverage(original, selected):
    expected = defaultdict(set)
    retained = defaultdict(set)
    for row in original:
        if row.get('directory_collection'):
            expected[row['directory_category']].update(row['directory_expected_ids'])
    for row in selected:
        if row.get('directory_collection'):
            retained[row['directory_category']].update(row['directory_member_ids'])
    return [{'category':category,'target_count':len(ids),'retained_count':len(retained[category] & ids),
             'missing_member_ids':sorted(ids-retained[category]),
             'status':'partial' if ids-retained[category] else 'complete',
             'scope':'directory_membership_not_full_document_review'} for category,ids in expected.items()]


def compact_directory_packets(rows, question='', goals=(), *, names_only=False):
    """Merge member lists, not facts. IDs/provenance remain backend-owned.

    Counts and other aggregates must not be reinterpreted as named members.
    Legacy packets without structured membership stay untouched.
    """
    import copy
    groups = defaultdict(list)
    for row in rows:
        if (row.get('directory_collection') and row.get('directory_members')
                and row.get('catalog_record_type') != 'directory_aggregate'):
            groups[row['directory_category']].append(row)
    replacements, removed = {}, set()
    intent = ' '.join([str(question), *map(str, goals)])
    for category, packets in groups.items():
        members = {m['id']:m for p in packets for m in p['directory_members']}
        result = copy.deepcopy(packets[0])
        lines=[]
        for member in members.values():
            name = str(member.get('name') or member['text'].split('；',1)[0])
            fields = member.get('attributes', {})
            labels = member.get('attribute_labels', {})
            suffix = '' if names_only else ''.join(
                '；'+label+'：'+str(fields[key]) for key,label in labels.items()
                if fields.get(key) not in (None,'') and
                    (label in intent or key in intent or key in result.get('collection_group_by',[])))
            lines.append(name+suffix)
        ids=list(members)
        result.update(directory_member_ids=ids, directory_item_count=len(ids),
            directory_expected_ids=list(dict.fromkeys(i for p in packets for i in p['directory_expected_ids'])),
            directory_members=list(members.values()),
            entity_coverage_ids=list(dict.fromkeys(i for p in packets for i in p.get('entity_coverage_ids',[]))),
            text=f'本轮{LABELS[category]}已登记成员清单，当前保留{len(ids)}项。仅支持名称及明示属性；未展示详情不表示没有资料。\n'+'\n'.join(lines),
            directory_compaction={'policy':'membership_first','detail_level':'names' if names_only else 'requested_attributes'},
            citations=list({json.dumps(r,ensure_ascii=False,sort_keys=True):r for p in packets for r in p.get('citations',[])}.values()))
        result['prepared_content_sha256']=hashlib.sha256(result['text'].encode()).hexdigest()
        # The merged packet is no longer one group's packet. Do not retain a
        # first-group count as though it described the complete member union.
        result.pop('collection_group_count', None)
        replacements[packets[0]['evidence_id']]=result
        removed.update(p['evidence_id'] for p in packets[1:])
    return [replacements.get(r['evidence_id'],r) for r in rows if r['evidence_id'] not in removed]


def trim_directory_member(row):
    """Last-resort partial list. Never claim omitted members were reviewed."""
    import copy
    if not row.get('directory_compaction') or len(row.get('directory_members', [])) < 2:
        return None
    result=copy.deepcopy(row)
    members=result['directory_members'][:-1]
    ids=[m['id'] for m in members]
    result.update(directory_members=members,directory_member_ids=ids,directory_item_count=len(ids),
        entity_coverage_ids=[i for i in row.get('entity_coverage_ids',[]) if i in ids],
        citations=list({json.dumps(c,ensure_ascii=False,sort_keys=True):c
            for m in members for c in m.get('citations',[])}.values()),
        text=f'本轮目录部分清单：保留{len(ids)}/{len(row["directory_expected_ids"])}项；其余因预算未展示，不表示不存在。\n'+
            '\n'.join(str(m.get('name') or m['text'].split('；',1)[0]) for m in members),
        directory_compaction={'policy':'membership_first','detail_level':'names','partial_reason':'minimum_membership_over_budget'})
    result['prepared_content_sha256']=hashlib.sha256(result['text'].encode()).hexdigest()
    return result


def collection_metadata(row):
    return {key:row[key] for key in ('directory_collection','directory_category',
        'directory_member_ids','directory_expected_ids','directory_members','directory_compaction','directory_item_count','fact_support_scope',
        'collection_action','collection_group_by','collection_total_count','collection_group_count','collection_missing_fields',
        'directory_level','directory_semantic_selection') if key in row}


def compact_coverage(coverage):
    return [{k:v for k,v in row.items() if k!='missing_member_ids'} |
            {'missing_count':len(row.get('missing_member_ids', []))} for row in coverage]


def coverage_notice(coverage, language='Chinese'):
    partial = [r for r in coverage if r['status']=='partial']
    if not partial:
        return ''
    english = str(language).casefold() in {'en','english'}
    details = '; '.join(f"{LABELS[r['category']]} {r['retained_count']}/{r['target_count']}" for r in partial)
    return ('Directory coverage is partial ('+details+'); omitted members are not evidence of absence.' if english
        else '目录覆盖范围：'+details+'。部分成员未进入本轮回答上下文；不能据此认定目录中不存在这些成员。')
