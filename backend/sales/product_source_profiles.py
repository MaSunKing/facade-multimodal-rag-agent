"""Reviewed source preference and exact color-model bindings, not new SKUs."""
import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[2]/'data/sales/config/product_source_profiles.json'


def rebuild_product_profiles(c, policy=None):
    policy = policy if policy is not None else json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {'catalog_product_attributes','knowledge_text_evidence'} <= tables:
        return {'profiles': 0}
    c.execute('CREATE TABLE IF NOT EXISTS catalog_product_source_profiles(product_id TEXT PRIMARY KEY REFERENCES products(product_id),profile_json TEXT NOT NULL)')
    c.execute('DELETE FROM catalog_product_source_profiles')
    public = {r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
    count = 0
    for pid, form, color, finish in c.execute('SELECT product_id,product_form,color_name,surface_finish FROM catalog_product_attributes').fetchall():
        profile = dict(policy.get('forms', {}).get(form, {}))
        if profile.get('source_id') not in public:
            continue
        profile.update(basis=policy['basis'], product_form=form, color_name=color,
                       registered_finish=finish, option_owner='color' if color else 'form',
                       boundary='体系可选项不代表已登记组合、完整一体板型号、图片、性能或项目采用关系')
        if policy.get('models_source_id') in public:
            matches = [m for m in policy.get('color_models', []) if m['color']==color and m['finish']==finish]
            if len(matches)>1:
                raise ValueError('ambiguous_color_finish_model')
            profile['registered_color_models'] = matches
            profile['models_source_id'] = policy['models_source_id']
        c.execute('INSERT INTO catalog_product_source_profiles VALUES(?,?)', (pid,json.dumps(profile,ensure_ascii=False)))
        count += 1
    return {'profiles':count}


def read_product_profile(c, product_id):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE name='catalog_product_source_profiles'").fetchone():
        return {}
    row = c.execute('SELECT q.profile_json FROM catalog_product_source_profiles q JOIN products p USING(product_id) WHERE q.product_id=? AND p.access_scope=?', (product_id,'public')).fetchone()
    if not row:
        return {}
    profile=json.loads(row[0])
    public={r[0] for r in c.execute("SELECT evidence_id FROM knowledge_text_evidence WHERE access_scope='public'")}
    if profile.get('source_id') not in public:
        return {}
    if profile.get('models_source_id') not in public:
        profile.pop('registered_color_models',None)
        profile.pop('models_source_id',None)
    return profile


def product_profile_text(c, product_id, *, include_parameters=True):
    p = read_product_profile(c, product_id)
    if not p:
        return ''
    parts=['目录默认采用已选PPT口径；指定来源/版本或检测样品时仍按对应资料，不修改其原文']
    if p.get('colors') and not p.get('color_name'):
        parts.append('体系饰面层花色选项：'+'、'.join(p['colors']))
    if p.get('cores'):
        parts.append('体系芯材选项：'+'、'.join(p['cores']))
    for model in p.get('registered_color_models', []):
        parts.append(f"本条目明确色型型号：{model['color']}／{model['finish']}／{model['code']}（不是完整一体板组合编码）")
    for field,label in [('standard_specification','PPT目录规格'),('panel_thickness','PPT目录面板厚度'),('weight_kg_m2','PPT目录面板重量kg/m²')]:
        if include_parameters and p.get(field):
            parts.append(label+'：'+p[field])
    parts.append(p['boundary'])
    return '；'.join(parts)


def product_profile_citations(c, product_id):
    p = read_product_profile(c, product_id)
    if not p or not c.execute("SELECT 1 FROM sqlite_master WHERE name='knowledge_evidence_sources'").fetchone():
        return []
    refs=[]
    for eid in [p['source_id'], p.get('models_source_id') if p.get('registered_color_models') else None]:
        if eid:
            refs.extend(json.loads(r[0]) for r in c.execute('SELECT reference_json FROM knowledge_evidence_sources WHERE evidence_id=?',(eid,)))
    return refs
