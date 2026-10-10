"""Six navigation domains; source authority and ACL are independent."""
from __future__ import annotations
import re
from collections import Counter

LABELS = {
    '01_standard_specification': '标准规范',
    '02_product_information': '产品资料',
    '03_construction_method': '施工工艺',
    '04_node_atlas': '节点图集',
    '05_project_case': '项目案例',
    '06_enterprise_information': '企业资料',
}
ALIASES = {'06_sales_playbook': '02_product_information',
           '05_project_cases': '05_project_case'}

# Navigation vocabulary, not entity names or a user-question keyword router.
DIRECTORY_CLASS_NAMES = ['产品', '项目', '工艺', '节点', '标准', '企业',
                         '产品系列', '产品目录', '项目目录',
                         'products', 'projects', 'processes', 'nodes', 'standards']

# Registered document/section topics are navigation classes, not entity names.
TOPICS = {
    '02_product_information':['产品介绍','目录规格','材料构成','产品对比','维护','检测报告'],
    '05_project_case':['项目应用','项目现场图'],
    '03_construction_method':['安装方法','施工步骤','基层处理','锚固要求'],
    '04_node_atlas':['转角节点','接缝节点','收口节点'],
    '01_standard_specification':['标准目录','技术条款'],
    '06_enterprise_information':['公司介绍','企业资质','专利','荣誉','新闻'],
}

CATEGORY_SCOPE = {
    '产品资料': '介绍/规格/材料/对比/维护，检测报告是产品子类，不是标准',
    '项目案例': '登记项目、应用和现场图；统计按可访问登记集合',
    '施工工艺': '安装方法、步骤、基层和锚固，不并入产品',
    '节点图集': '构造图、接缝和收口等说明，不等于施工全文',
    '标准规范': '标准名称和条款；目录可回答有哪些，条款含义需正文',
    '企业资料': '身份、资质、专利、荣誉和新闻',
}


def canonical_domains(values):
    return list(dict.fromkeys(ALIASES.get(x, x) for x in values if ALIASES.get(x, x) in LABELS))


def planner_capability_map():
    """Small backend-authored capability map, no corpus content or private IDs."""
    return {'knowledge_categories': LABELS,
            'document_topics': TOPICS,
            'category_scope': dict(CATEGORY_SCOPE),
            'operations': ['lookup', 'summary', 'comparison'],
            'source_roles': {'company_rag':'企业六类资料、SQL目录/关系和库内证据',
                'customer_documents':'本会话客户附件；不自动写入企业库',
                'public_web_search':'公开和实时资料；不能替代私有公司资料',
                'visual_inspection':'当前客户直接上传图的按需识别；公司图库永不运行时看原图',
                'general_chat':'无需外部取证的一般交流'},
            'directory': '产品、项目实体目录；其他类别按文档、章节和主题定位',
            'relations': ['产品↔项目', '实体↔图文', '体系↔施工/节点/技术参考', '证据↔来源位置'],
            'boundary': '关联用于导航，不证明型号性能或项目合规；权限由后端控制'}


def audit_capability_map():
    """Shared vocabulary, not a claim that any particular source exists."""
    return {**planner_capability_map(),
        'support_boundary': '能力地图不证明库完整。名称目录无需全文才能列举；用户问内容/参数才需正文。实际证据与覆盖状态决定支持。'}


def classify_fragment(row):
    old = row.get('original_knowledge_domains', row.get('knowledge_domains')) or []
    domains = list(dict.fromkeys(ALIASES.get(x, x) for x in old if ALIASES.get(x, x) in LABELS))
    categories = set(row.get('document_categories') or [])
    visual = row.get('visual') or {}
    catalog = row.get('catalog') or visual.get('catalog') or {}
    heading = ' '.join(str(s.get('section_heading') or '') for s in row.get('source_refs', []))
    text = str(row.get('text') or row.get('search_text') or '')
    # Headings / literal fragment content, never a whole mixed document title.
    lead = heading + ' ' + str(row.get('customer_caption') or '') + ' ' + text[:180]
    technical = set(domains) & {'01_standard_specification','03_construction_method','04_node_atlas'}
    primary = None
    subtype = None
    if row.get('kind') == 'project_case' or 'official_website_project_case' in categories or (row.get('gallery_type') or catalog.get('gallery_type')) == 'project_case' or re.search(r'^\s*(应用案例|山东省部分案例)', lead):
        primary = '05_project_case'
    elif re.search(r'检测报告|送检样品|检验报告|检测项目|抗冻融性|抗冲击性', lead) and not technical:
        primary, subtype = '02_product_information', 'test_report_or_performance'
    elif 'enterprise_identity_profile' in categories or re.search(
        r'企业荣誉|公司简介|公司介绍|企业简介|企业概况|专利证书|发明专利|实用新型专利|软件著作权|科技查新|查新报告|科技成果|高新技术企业|展会|访谈节目|质量奖|营业执照', lead):
        # Product comparison source remains a product comparison even if it
        # mentions the manufacturer's credentials.
        if 'internal_product_comparison' not in categories and not technical:
            primary = '06_enterprise_information'
    elif not technical and re.search(r'节点详图|阴角节点|阳角节点|门窗洞口节点|收口节点', lead):
        primary = '04_node_atlas'
    elif not technical and re.search(r'施工步骤|安装步骤|施工流程|施工工艺流程', lead):
        primary = '03_construction_method'
    if primary:
        domains = [primary]
    if not domains:
        domains = ['05_project_case'] if row.get('kind') == 'project_case' else []
    if 'internal_product_comparison' in categories:
        subtype = 'product_comparison'
    result = {'knowledge_domains': domains, 'classification_basis':
              'fragment_content_or_structured_source' if primary else 'canonical_source_domain',
              'classification_status': 'classified' if domains else 'needs_review'}
    if subtype:
        result['knowledge_subtype'] = subtype
    return result


def normalize_documents(documents):
    for row in documents:
        if 'original_knowledge_domains' not in row:
            row['original_knowledge_domains'] = list(row.get('knowledge_domains') or [])
        row.update(classify_fragment(row))
    return dict(Counter(domain for row in documents for domain in row['knowledge_domains']))
