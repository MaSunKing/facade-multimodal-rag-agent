"""Resolve directory entities and expand navigational SQL edges, not facts."""
import json
import re
import sqlite3


def category_candidate_ids(database, categories):
    """Public classified Evidence pool; categories never grant access."""
    from backend.sales.content_taxonomy import canonical_domains
    domains = set(canonical_domains(categories))
    if not domains or not database.is_file():
        return set()
    with sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'knowledge_text_evidence' not in tables:
            return set()
        if 'metadata_json' not in {r[1] for r in c.execute('PRAGMA table_info(knowledge_text_evidence)')}:
            return set()
        ids = {eid for eid,metadata in c.execute(
            "SELECT evidence_id,metadata_json FROM knowledge_text_evidence WHERE access_scope='public'")
            if domains & set(canonical_domains(json.loads(metadata or '{}').get('knowledge_domains') or []))}
        if {'visual_evidence','visual_assets'} <= tables:
            ids.update('visual:'+aid for aid,eid in c.execute('''SELECT v.asset_id,v.evidence_id FROM visual_evidence v
                JOIN visual_assets a USING(asset_id) WHERE a.access_scope='public' ''') if eid in ids)
        return ids


def normalized(value):
    return re.sub(r'[\s®·：:＋+\-—_（）()]', '', value or '').lower()


def entity_names(row, project=False):
    name = row['project_name'] if project else row['name']
    names = {normalized(name)}
    names.update(normalized(alias) for alias in row.get('aliases', []) if isinstance(alias,str))
    if project:
        # Retain short project references after omitted geographic prefixes.
        full = normalized(name)
        names.update(full[-size:] for size in range(5,min(len(full),12)+1))
        for location in (row.get('province'),row.get('city')):
            location = normalized(location)
            for suffix in ('省','市','自治区'):
                location = location.removesuffix(suffix)
            if location:
                names |= {n.removeprefix(location) for n in list(names)}
    else:
        finish = normalized(row.get('surface_finish'))
        if finish:
            names.add(normalized(name).removesuffix(finish))
    return {n for n in names if len(n)>=(4 if project else 2)}


def resolve_relation_scope(database, query, *, categories=(), targets=(), filters=None, operation='unknown', scorer=None):
    result={'status':'unresolved','entity_ids':[],'candidate_ids':[], 'relations':[], 'evidence_groups':[],
            'boundary':'Related documents are references, not project-specific installation or compliance proof.'}
    if not database.is_file():return result
    with sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'products','project_cases','knowledge_text_evidence','knowledge_evidence_sources'}<=tables:return result
        question=normalized(query)
        products=[dict(r) for r in c.execute("SELECT * FROM products WHERE access_scope='public'")]
        cases=[dict(r) for r in c.execute("SELECT * FROM project_cases WHERE access_scope='public'")]
        if 'project_name_aliases' in tables:
            aliases = {}
            for identity, name in c.execute('SELECT case_id,name FROM project_name_aliases'):
                aliases.setdefault(identity, []).append(name)
            for case in cases:
                case['aliases'] = aliases.get(case['case_id'], [])
        public_evidence={r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
        public_documents={r[0] for r in c.execute('''SELECT DISTINCT s.document_key FROM knowledge_evidence_sources s
          JOIN knowledge_text_evidence e USING(evidence_id) WHERE e.access_scope='public' ''')}
        matched_families = set()
        if 'technical_family_aliases' in tables:
            public_families={p['family'] for p in products}
            family_aliases=[(a,f) for a,f in c.execute('SELECT alias,family FROM technical_family_aliases')
                            if f in public_families]
            matched_families = {family for alias,family in family_aliases
                                if normalized(alias) and normalized(alias) in question}
            # Registered suffixes support omitted name qualifiers only when
            # they uniquely identify a family. Short generic fragments do not.
            suffixes={}
            for alias,family in family_aliases:
                value=normalized(alias)
                for n in range(3,len(value)):
                    suffixes.setdefault(value[-n:],set()).add(family)
            for suffix, owners in suffixes.items():
                if len(owners)==1 and suffix in question:
                    matched_families.update(owners)
        product_pool = [p for p in products if p['family'] in matched_families] if matched_families else products
        from backend.sales.entity_directory import resolve_entities
        selected_products,selected_cases,directory=resolve_entities(product_pool,cases,query,
            categories=categories,targets=targets,filters=filters,operation=operation,scorer=scorer)
        # A registered family is an object level, not an instruction to make
        # every child color/core variant a separate answer requirement.
        product_ids = {p['product_id'] for p in product_pool}
        family_level = bool(matched_families and not any(
            m['entity_id'] in product_ids and m['basis'] in {'exact_name','registered_name_or_catalog_variant'}
            for m in directory['matches']))
        family_targets = [{'id':'family:'+family,'name':family,'kind':'family',
            'aliases':list(dict.fromkeys([*[a for a,f in family_aliases if f==family],
                *[s for s,owners in suffixes.items() if owners=={family} and s in question]]))}
            for family in sorted(matched_families)] if family_level else []
        if family_level:
            selected_products = product_pool
            directory.update(status='collection', family_filter=sorted(matched_families),
                target_level='family',
                matches=[{'entity_id':t['id'],'basis':'registered_family_alias'} for t in family_targets])
        if matched_families:
            directory['family_filter'] = sorted(matched_families)
        result['directory_resolution']=directory
        result['entity_targets'] = (family_targets if family_level else [{'id':r['product_id'],'name':r['name'],'kind':'entity','aliases':sorted(entity_names(r))} for r in selected_products]) + [
            {'id':r['case_id'],'name':r['project_name'],'kind':'entity','aliases':sorted(entity_names(r,True))} for r in selected_cases]
        result['directory_member_ids'] = [p['product_id'] for p in selected_products] if family_level else []
        from backend.sales.catalog_structure import read_product_attributes
        result['product_attributes']=read_product_attributes(c,[p['product_id'] for p in selected_products])
        ids=set(); families=set(matched_families); document_keys=set()
        # Reviewed family aliases resolve a product form, never a color-specific
        # method or an invented SKU. Existing exact entity matches remain intact.
        if 'technical_family_aliases' in tables:
            public_families={p['family'] for p in products}
            for alias,family in c.execute('SELECT alias,family FROM technical_family_aliases'):
                if family in public_families and normalized(alias) in question:
                    families.add(family)
        for table, rows, key, edges, image_edges in (
            ('product',selected_products,'product_id','product_evidence','product_visuals'),
            ('project',selected_cases,'case_id','project_evidence','project_case_visuals')):
            for row in rows:
                eid=row[key]
                if not (family_level and table=='product'):
                    result['entity_ids'].append(eid)
                ids.add(eid)
                if edges in tables:
                    related=[r[0] for r in c.execute(f'SELECT evidence_id FROM {edges} WHERE {key}=?',(eid,)) if r[0] in public_evidence]
                    ids.update(related)
                    result['evidence_groups'].append({'relation':'entity_source','entity_id':eid,'ids':related,
                        **({'family':row['family']} if table=='product' else {})})
                if image_edges in tables:
                    ids.update('visual:'+r[0] for r in c.execute(f'SELECT asset_id FROM {image_edges} WHERE {key}=?',(eid,)))
                if table=='product':
                    families.add(row['family'])
                    if 'project_product_links' in tables:
                        for linked in c.execute('''SELECT l.case_id FROM project_product_links l JOIN project_cases p USING(case_id)
                          WHERE l.product_id=? AND p.access_scope='public' ''',(eid,)):
                            ids.add(linked[0])
                            result['relations'].append({'relation':'product_project_reference','product_id':eid,'case_id':linked[0]})
                            if 'project_evidence' in tables:
                                related=[r[0] for r in c.execute('SELECT evidence_id FROM project_evidence WHERE case_id=?',(linked[0],)) if r[0] in public_evidence]
                                ids.update(related)
                                result['evidence_groups'].append({'relation':'product_project_reference','entity_id':eid,'ids':related})
                            if 'project_case_visuals' in tables:
                                ids.update('visual:'+r[0] for r in c.execute('SELECT asset_id FROM project_case_visuals WHERE case_id=?',(linked[0],)))
                elif 'project_product_links' in tables:
                    for p in c.execute('''SELECT p.product_id,p.family FROM project_product_links l
                      JOIN products p USING(product_id) WHERE l.case_id=? AND p.access_scope='public' ''',(eid,)):
                        families.add(p['family'])
                        if 'product_evidence' in tables:
                            related=[r[0] for r in c.execute('SELECT evidence_id FROM product_evidence WHERE product_id=?',(p['product_id'],)) if r[0] in public_evidence]
                            ids.update(related)
                            result['evidence_groups'].append({'relation':'project_product_reference','entity_id':eid,'ids':related})
                        if 'product_visuals' in tables:
                            ids.update('visual:'+r[0] for r in c.execute('SELECT asset_id FROM product_visuals WHERE product_id=?',(p['product_id'],)))
                    # Supplied catalogue form may be more explicit than older SKU edges.
                    if '保温' in (row.get('product_name') or ''):
                        families.update(p['family'] for p in products if p.get('insulation_type'))
        from backend.sales.node_relations import related_node_assets
        node_assets=related_node_assets(c,query,families)
        for node in node_assets:
            ids.add('visual:'+node['asset_id'])
            result['relations'].append(node)
            result['evidence_groups'].append({'relation':'system_node_reference','ids':['visual:'+node['asset_id']]})
        from backend.sales.content_taxonomy import canonical_domains
        domains=set(canonical_domains(categories))
        # Topic categories are independent of entity categories. A selected
        # product must not suppress enterprise/standard/process/node directories.
        topic_domains=domains-{'02_product_information','05_project_case'}
        unresolved=not result['entity_ids'] and not families and not node_assets
        if unresolved or (operation == 'summary' and not targets and topic_domains):
            domains=domains if unresolved else topic_domains
            if domains and 'metadata_json' in {r[1] for r in c.execute('PRAGMA table_info(knowledge_text_evidence)')}:
                for row in c.execute("SELECT evidence_id,metadata_json FROM knowledge_text_evidence WHERE access_scope='public'"):
                    metadata=json.loads(row['metadata_json'] or '{}')
                    if domains & set(canonical_domains(metadata.get('knowledge_domains') or [])):
                        ids.add(row['evidence_id'])
                result['topic_directory']={'categories':sorted(domains),'candidate_count':len(ids),'basis':'section_evidence_category'}
                if 'visual_evidence' in tables:
                    topic_text_ids=set(ids)
                    ids.update('visual:'+r[0] for r in c.execute('SELECT asset_id,evidence_id FROM visual_evidence') if r[1] in topic_text_ids)
            if not ids:return result
        from backend.sales.technical_sections import related_sections
        from backend.sales.technical_sections import related_project_sections
        for section in related_project_sections(c,[x['case_id'] for x in selected_cases]):
            ids.update(section['evidence_ids'])
            ids.update('visual:'+aid for aid in section['asset_ids'])
            result['relations'].append({'relation':'source_method_reference','section_id':section['section_id'],
                'case_id':section['case_id'],'provenance':section['provenance']})
            result['evidence_groups'].append({'relation':'source_method_reference','section_id':section['section_id'],
                'ids':section['evidence_ids']})
        for section in related_sections(c,families):
            ids.update(section['evidence_ids'])
            ids.update('visual:'+aid for aid in section['asset_ids'])
            result['relations'].append({'relation':section['relation'],'section_id':section['section_id'],
                'family':section['family'],'provenance':section['provenance']})
            result['evidence_groups'].append({'relation':section['relation'],'section_id':section['section_id'],
                'family':section['family'],'ids':section['evidence_ids']})
        if 'family_document_links' in tables:
            for family in sorted(families):
                for link in c.execute('SELECT * FROM family_document_links WHERE family=?',(family,)):
                    proof=json.loads(link['provenance_json'])
                    if link['document_key'] not in public_documents or proof.get('evidence_id') not in public_evidence:
                        continue
                    document_keys.add(link['document_key'])
                    result['relations'].append({'relation':link['relation'],'family':family,
                      'document_key':link['document_key'],'provenance':proof})
        if 'document_reference_links' in tables:
            for key in list(document_keys):
                for link in c.execute('SELECT * FROM document_reference_links WHERE from_document=?',(key,)):
                    if link['to_document'] not in public_documents or link['evidence_id'] not in public_evidence:
                        continue
                    document_keys.add(link['to_document'])
                    result['relations'].append({'relation':'standard_reference','document_key':link['to_document'],
                      'reference_code':link['reference_code'],'provenance':json.loads(link['provenance_json'])})
        for key in document_keys:
            related=[r[0] for r in c.execute('SELECT evidence_id FROM knowledge_evidence_sources WHERE document_key=?',(key,)) if r[0] in public_evidence]
            ids.update(related)
            relation=next((l['relation'] for l in result['relations'] if l.get('document_key')==key),'reference')
            result['evidence_groups'].append({'relation':relation,'document_key':key,'ids':related})
            if 'visual_document_links' in tables:
                ids.update('visual:'+r[0] for r in c.execute('SELECT asset_id FROM visual_document_links WHERE document_key=?',(key,)))
        if family_level:
            result['entity_ids'].extend(t['id'] for t in family_targets)
            for target in family_targets:
                family_ids={eid for g in result['evidence_groups'] if g.get('family')==target['name'] for eid in g['ids']}
                result['evidence_groups'].append({'relation':'family_reference','entity_id':target['id'],
                    'family':target['name'],'ids':sorted(family_ids)})
        # ACL and existence are enforced after traversing every edge, not just seeds.
        permitted={r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
        permitted.update(p['case_id'] for p in cases)
        if 'visual_assets' in tables:
            permitted.update('visual:'+r[0] for r in c.execute("SELECT asset_id FROM visual_assets WHERE access_scope='public'"))
        if 'visual_retrieval_review' in tables:
            permitted.difference_update('visual:'+r[0] for r in c.execute("SELECT asset_id FROM visual_retrieval_review WHERE status='excluded'"))
        result.update(status='directory_matched',candidate_ids=sorted(ids & permitted))
    return result
