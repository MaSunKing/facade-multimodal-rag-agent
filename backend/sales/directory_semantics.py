"""Semantic directory views, not SKU creation or factual classification."""
import math
from collections import defaultdict

VIEWS={
    'systems':'产品体系、板材种类和整体介绍；有哪些产品，单板与保温一体板的层级',
    'colors':'花色属性选项；颜色、色系、有哪些颜色',
    'finishes':'饰面属性选项；表面效果、纹理、光面荔枝面哑光面',
    'cores':'保温芯材属性选项；板里面填什么、内部保温材料',
    'models':'已登记型号与色型清单；型号编码，具体款式，全部型号',
    'composition':'材料构成和结构层次；板由什么组成，复合结构',
    'parameters':'具体参数资料；厚度、重量、尺寸和规格',
}

def locate_view(question, *, scorer=None):
    from backend.sales.entity_directory import rank_directory
    from backend.request_budget import RequestBudgetExceeded
    keys=list(VIEWS)
    try:
        scores=list((scorer or rank_directory)(question,list(VIEWS.values())))
        if len(scores)!=len(keys) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError('directory_view_scores_invalid')
        if max(scores)-min(scores)<1e-6:
            raise ValueError('directory_views_indistinguishable')
        ranked=sorted(zip(scores,keys),reverse=True)
        # Keep the two relevant navigation alternatives. These are ranking
        # nominations, never calibrated confidence or hard evidence filters.
        return {'views':[k for _,k in ranked[:2]],'status':'candidate_views',
                'ranked_candidates':[{'view':k,'score':float(s)} for s,k in ranked],
                'method':'existing_reranker','semantic_support_verified':False}
    except RequestBudgetExceeded:
        raise
    except Exception as exc:
        return {'views':['systems','colors','finishes','cores'],'status':'broad_navigation_fallback',
                'method':'directory_structure','error_type':type(exc).__name__,
                'semantic_support_verified':False}

def hierarchy_members(members, selection, profiles):
    groups=defaultdict(list)
    for m in members:groups[m['attributes'].get('family') or '未登记体系'].append(m)
    views=set(selection['views'])
    output=[]
    for family, rows in sorted(groups.items()):
        attrs=[r['attributes'] for r in rows]
        ps=[profiles.get(r['id'],{}) for r in rows]
        def values(field, profile_field=None):
            result={str(a[field]) for a in attrs if a.get(field)}
            if profile_field:
                result.update(str(v) for p in ps for v in p.get(profile_field,[]) if v)
            return sorted(result)
        text=family+'（体系）'
        # Broad questions get concise options, not every registered SKU.
        broad=selection['views'][0]=='systems'
        for view, field, pfield, label in [
            ('colors','color_name','colors','花色选项'),
            ('cores','insulation_core','cores','保温芯材选项')]:
            vs=values(field,pfield)
            if vs and (broad or view in views):text+='\n  '+label+'：'+'、'.join(vs)
        finishes=sorted({v for a in attrs for v in a.get('available_surface_finishes',[])}
                        | {a['surface_finish'] for a in attrs if a.get('surface_finish')})
        if finishes and (broad or 'finishes' in views):text+='\n  饰面选项：'+'、'.join(finishes)
        if 'models' in views and not broad:
            for row,p in zip(rows,ps):
                codes=[m.get('code') for m in p.get('registered_color_models',[]) if m.get('code')]
                text+='\n  已登记条目：'+row['name']+('；型号：'+'、'.join(codes) if codes else '；未附明确型号编码')
        text+='\n  选项不是已登记组合；具体型号、参数和构造须用关联正文取证。'
        output.append({'id':'family:'+family,'name':text,'text':text,
            'attributes':{'family':family,'category':'02_product_information'},
            'child_ids':[r['id'] for r in rows],
            'citations':[ref for r in rows for ref in r.get('citations',[])]})
    return output
