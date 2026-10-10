"""Deterministic source-label navigation for node assets, not engineering proof."""
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

POLICY_PATH=Path(__file__).resolve().parents[2]/'data/sales/node_navigation_policy.json'

def installation_defaults():
    config=json.loads(POLICY_PATH.read_text(encoding='utf8')) if POLICY_PATH.is_file() else {}
    defaults=config.get('installation_defaults',{})
    if any(form not in FORM_FAMILIES or method not in {'干挂','粘锚'} for form,method in defaults.items()):
        raise ValueError('invalid_node_installation_defaults')
    return defaults

FORM_FAMILIES={'decorative_panel':'真岩®无机仿石装饰板','insulated_panel':'真岩®保温装饰一体板'}
KINDS={'阴角':('阴角',),'阳角':('阳角',),'门窗收口':('门窗','窗洞','窗台','窗上','窗下','洞口'),
       '女儿墙':('女儿墙',),'防火隔离':('防火隔离','防火带'),
       '锚固':('锚固','锚栓','锚钉'),'变形缝':('沉降缝','沉降','伸缩缝','变形缝'),
       '接缝':('平缝','接缝','拼缝'),'材料构成':('材料组成','系统构成','材料构成'),
       '性能指标表':('性能指标','性能要求','性能表','厚度对照','规格尺寸'),
       '施工流程':('施工工艺流程','施工流程')}

def forms(text):
    found=[]
    if '结构一体化' in text:
        found.append('structural_integrated')
    elif '一体板' in text or '保温装饰板' in text:
        found.append('insulated_panel')
    if '单板' in text:
        found.append('decorative_panel')
    return found

def methods(text):
    result=[]
    for value,terms in [('干挂',('干挂','穿透支承')),('粘锚',('粘锚',))]:
        if any(term in text and not re.search(r'(?:不能|不得|不适用|禁止|不采用|非)[^，。；\n]{0,4}'+re.escape(term),text) for term in terms):
            result.append(value)
    # Mere adhesion/strip bonding is not proof of a full bonded-anchor system.
    return result

def kinds(text):
    matches=[kind for kind,terms in KINDS.items() if any(term in text for term in terms)]
    if '性能指标表' in matches and not any(term in text for term in ('节点','构造','示意')):
        return ['性能指标表']
    return matches

def source_key(row,metadata):
    value=metadata.get('source_pdf') or metadata.get('source_path') or metadata.get('document_id')
    return (re.sub(r'[/\\]+','/',str(value or row['asset_id'])).casefold(),row['source_page'],row['access_scope'])

def rebuild_node_relations(c):
    tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {'visual_assets','visual_directory'} <= tables:return {}
    columns={r[1] for r in c.execute('PRAGMA table_info(visual_assets)')}
    if 'access_scope' not in columns:
        # Legacy catalogues may be classified, but must not gain cross-asset
        # navigation without explicit ownership boundaries. Preserve old links.
        return {'state':'skipped_missing_access_scope'}
    c.execute('''CREATE TABLE IF NOT EXISTS node_asset_relations(
        asset_id TEXT NOT NULL REFERENCES visual_assets(asset_id), product_form TEXT NOT NULL,
        installation_method TEXT NOT NULL,node_kind TEXT NOT NULL,provenance_json TEXT NOT NULL,
        PRIMARY KEY(asset_id,product_form,installation_method,node_kind))''')
    rows=[];peers=defaultdict(list)
    for row in c.execute('''SELECT a.asset_id,a.title,a.document_name,a.source_page,a.access_scope,a.metadata_json
        FROM visual_assets a JOIN visual_directory d USING(asset_id) WHERE d.category='node' '''):
        aid,title,name,page,scope,raw=row
        item=dict(asset_id=aid,title=title or '',document_name=name or '',source_page=page,access_scope=scope)
        m=json.loads(raw or '{}')
        own=[item['title'],m.get('section_heading') or '',m.get('caption') or '']
        own.extend(x.get('title','') for x in m.get('technical_occurrences',[]))
        label='；'.join(dict.fromkeys(x for x in own if x))
        key=source_key(item,m)
        peers[key].append((aid,label))
        rows.append((item,m,key,label))
    c.execute('DELETE FROM node_asset_relations')
    for item,m,key,label in rows:
        # Whole-page assets may inherit labels from same-source, same-page,
        # same-ACL cropped diagrams, never nearby pages or another document.
        shared=[dict(asset_id=aid,label=text) for aid,text in peers[key] if aid!=item['asset_id'] and '完整原页' not in text]
        text=label
        if '完整原页' in label and shared:
            text+='；'+'；'.join(x['label'] for x in shared)
        product_forms=forms(text)
        basis='asset_labels'
        if not product_forms:
            product_forms=forms(item['document_name'])
            basis='source_document_scope' if product_forms else 'general_reference'
        product_forms=product_forms or ['general']
        explicit_methods=methods(text)
        selected_kinds=kinds(text) or ['通用构造页']
        proof=dict(source_document=item['document_name'],source_page=item['source_page'],
            source_identity=key[0],label=label,same_page_labels=shared if '完整原页' in label else [],
            product_form_basis=basis,method_basis='explicit_label_only',
            relation='system_node_reference',claim_boundary='体系导航参考，不证明具体项目使用、颜色专属或工程合规')
        for form in product_forms:
            default=installation_defaults().get(form,'')
            prohibited=bool(default and re.search(r'(?:不能|不得|不适用|禁止|不采用|非)[^，。；\n]{0,4}'+re.escape(default),text))
            selected_methods=explicit_methods or ([default] if default and not prohibited else [''])
            for method in selected_methods:
                for kind in selected_kinds:
                    relation_proof={**proof,'method_basis':('explicit_label_only' if explicit_methods else
                        'user_confirmed_catalog_default' if method else 'unspecified'),
                        'default_policy_version':1 if method and not explicit_methods else None}
                    c.execute('INSERT INTO node_asset_relations VALUES(?,?,?,?,?)',
                        (item['asset_id'],form,method,kind,json.dumps(relation_proof,ensure_ascii=False)))
    return {'assets':len(rows),'relationships':c.execute('SELECT count(*) FROM node_asset_relations').fetchone()[0],
        'forms':dict(c.execute('SELECT product_form,count(DISTINCT asset_id) FROM node_asset_relations GROUP BY product_form')),
        'methods':dict(c.execute("SELECT installation_method,count(DISTINCT asset_id) FROM node_asset_relations WHERE installation_method<>'' GROUP BY installation_method")),
        'explicit_methods':dict(c.execute("SELECT installation_method,count(DISTINCT asset_id) FROM node_asset_relations WHERE json_extract(provenance_json,'$.method_basis')='explicit_label_only' GROUP BY installation_method")),
        'defaulted_assets':c.execute("SELECT count(DISTINCT asset_id) FROM node_asset_relations WHERE json_extract(provenance_json,'$.method_basis')='user_confirmed_catalog_default'").fetchone()[0]}

def related_node_assets(c,query,families=()):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='node_asset_relations'").fetchone():return []
    wanted_kinds=kinds(query); wanted_methods=methods(query); wanted_forms=forms(query)
    if not wanted_kinds and not any(term in query for term in ('节点','构造图','图集')):
        return []
    if not wanted_forms:
        wanted_forms=[form for form,family in FORM_FAMILIES.items() if family in families]
    excluded=set()
    if c.execute("SELECT 1 FROM sqlite_master WHERE name='visual_retrieval_review'").fetchone():
        excluded={r[0] for r in c.execute("SELECT asset_id FROM visual_retrieval_review WHERE status='excluded'")}
    result=[]
    for aid,form,method,kind,proof in c.execute('''SELECT n.* FROM node_asset_relations n
        JOIN visual_assets a USING(asset_id) WHERE a.access_scope='public' ORDER BY n.asset_id,n.node_kind'''):
        if aid in excluded or (wanted_forms and form not in wanted_forms):continue
        if wanted_kinds and kind not in wanted_kinds:continue
        # Unknown method does not become a specific-method hit.
        if wanted_methods and method not in wanted_methods:continue
        result.append(dict(asset_id=aid,product_form=form,installation_method=method or None,
            node_kind=kind,relation='system_node_reference',provenance=json.loads(proof)))
    return result
