"""Build a versioned 100-case behaviour benchmark and deterministic fixtures.

No generated gold company facts: company/web cases require evidence review.
Synthetic attachment gold is explicit. Runtime output is not its own gold.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/sales/evaluation/agent_eval_100_v1'


COMPANY = {
 '产品资料': [
 '你们有哪些产品系列？', '你们单板有哪些花色？', '介绍一下保温一体板并给我图片。',
 '黄金麻有哪些已登记的饰面？', '单板和保温一体板有什么区别？', '黄金麻和白麻的饰面分别有哪些？',
 '介绍一下真岩石的材料构成。', '产品资料里有哪些检测报告？说明适用型号。',
 '介绍黄金麻的特点、安装方法和适用场景。', '只给我白麻样板图，不需要分析图片。'],
 '项目案例': [
 '你们有哪些项目案例？', '项目案例按省份统计各有多少个？未登记省份单列。',
 '山东有哪些项目？', '哪些项目使用了保温一体板？', '哪些项目使用单板？',
 '给我两个项目的现场图，并说明分别属于哪个项目。',
 '项目采用的产品和安装方法分别是什么？', '按城市整理已登记项目；目录不完整请说明覆盖范围。'],
 '施工工艺': [
 '介绍单板干挂施工步骤。', '保温一体板粘锚安装需要哪些基层条件？',
 '干挂和粘锚的安装方法有什么区别？', '施工前有哪些准备和检查要求？',
 '锚固件有哪些安装要求？请保留适用条件。', '介绍接缝处理和后续维护要求。'],
 '节点图集': [
 '给我门窗洞口收口节点图。', '单板转角节点有哪些？', '保温一体板接缝节点怎么处理？',
 '给我干挂锚固节点的图和说明。', '节点图集按节点类型整理一下。', '比较单板和一体板的收口节点。'],
 '标准规范': [
 '库里有哪些标准规范？', '外墙保温装饰板适用哪些技术要求？',
 '检测报告和标准规范分别有哪些？不要混为一类。',
 '防火性能要求有哪些适用条件？', '不同版本的技术要求有冲突吗？说明来源和版本。'],
 '企业资料': [
 '你们是谁，主营什么？', '企业有哪些资质和证书？', '介绍企业专利和荣誉。',
 '企业资料中有哪些新闻？按资料日期整理。', '企业介绍和产品介绍分开概括。']}

ATTACHMENTS = [
 ('概括方案A。', ['a.txt'], ['A板','18毫米','干挂']),
 ('方案A厚度多少？', ['a.txt'], ['18']),
 ('方案A安装方法是什么？', ['a.txt'], ['干挂']),
 ('方案A的价格是否含税？', ['a.txt'], ['120','含税']),
 ('方案A有哪些使用限制？', ['a.txt'], ['室内']),
 ('比较方案A和方案B的厚度。', ['a.txt','b.txt'], ['18','25']),
 ('分别介绍方案A和方案B安装方法。', ['a.txt','b.txt'], ['干挂','粘锚']),
 ('比较两方案的价格口径。', ['a.txt','b.txt'], ['含税','不含税']),
 ('比较两份文件的适用场景和限制。', ['a.txt','b.txt'], ['室外','室内']),
 ('两份方案付款条件分别是什么？', ['a.txt','b.txt'], ['30','40']),
 ('表格中2026-10-08甲厂产量是多少？', ['records.csv'], ['20']),
 ('分别列出甲厂和乙厂10月8日的产量。', ['records.csv'], ['20','30']),
 ('统计表格中10月8日两厂总产量并说明单位。', ['records.csv'], ['50','吨']),
 ('表格最后一行是否可以作为10月8日产量？', ['records.csv'], ['10月9','10月8']),
 ('图片中有哪些几何图形？', ['shapes.png'], ['红','蓝']),
 ('比较图片中两个图形的颜色和形状。', ['shapes.png'], ['红','蓝']),
 ('能根据这张图确定材料防火等级吗？', ['shapes.png'], []),
 ('概括长文件主要内容，并说明覆盖范围。', ['long.txt'], []),
 ('长文件最后的禁止事项是什么？', ['long.txt'], ['禁止浸水']),
 ('比较长文件和方案A的安装条件，不要声称全文已完整审阅。', ['long.txt','a.txt'], ['干挂'])]

WEB = [
 '解释地球公转与自转的区别，引用公开来源。', '查一下中国国家图书馆的官网和服务介绍。',
 '今天新加坡天气如何？', '今天温州天气如何？', '分别查今天新加坡和大连的天气。',
 '查昨天与今天北京的天气，区分两天。', '新加坡明天天气预报如何？不要当成实测。',
 '2024年10月8日与今天的数据应如何区分？查公开资料说明。',
 '截至今天，查最新的外墙保温相关标准版本。', '查2024年发布的外墙保温相关规范。',
 '访问https://www.weather.gov.sg/，说明能确认什么天气信息。',
 '查北京、上海、杭州今天的天气，分别给来源。', '查上海今天气温与明天预报，不能混用日期。',
 '查一种材料的防火性能，区分厂家宣传与检测依据。', '查公开资料解释粘锚和干挂，不替代公司私有项目事实。']

TURNS = [
 (['查黄金麻厚度。','再查白麻。'], '追加对象，两个对象均保留'),
 (['查黄金麻厚度。','不用黄金麻，改成白麻。'], '替换对象，只保留白麻'),
 (['介绍单板。','再加上安装方法。'], '追加维度'),
 (['比较单板和一体板的厚度。','不看厚度，改比较施工方法。'], '替换维度'),
 (['查黄金麻。','我说错了，是黄金钻。'], '纠正名称'),
 (['查新加坡今天的天气。','再加大连。'], '两目标原子查询'),
 (['查新加坡今天的天气。','再加温州。','再加大连。'], '三次要求均保留'),
 (['查新加坡今天的天气。','不用了，改查温州。'], '联网替换，不查旧目标'),
 (['比较方案A和方案B。','只看方案A的付款条件。'], '切换到局部查找'),
 (['概括方案A。','改为比较A和B安装方法。'], '任务类型更新'),
 (['介绍公司产品。','王者荣耀你知道吗？'], '换话题，不强制企业RAG'),
 (['介绍单板。','只给图片。'], '图片展示只使用标签'),
 (['介绍黄金麻。','它有哪些饰面？'], '指代，仅必要历史辅助'),
 (['查昨天北京天气。','改成今天。'], '替换日期，旧时间失效'),
 (['查山东项目。','江苏呢？'], '对象省略与范围识别'),
 (['介绍单板和一体板。','不要一体板。'], '删除目标'),
 (['给我黄金麻图片。','再加白麻。'], '追加图片目标'),
 (['介绍项目案例。','按省份统计。'], '集合统计覆盖'),
 (['概括方案A。','再加方案B。'], '已解析附件复用'),
 (['查公司产品。','只查公开网络资料。'], '替换来源，不混私有资料')]


def cases():
    rows=[]
    def add(group, question, **extra):
        rows.append({'id':f'AG100-{len(rows)+1:03d}', 'group':group, 'question':question,
            'timeout_seconds':1200, 'rubric':['保留用户有效要求','来源与证据支持一致',
                '不足时说明具体缺口，不捏造','格式与引用ID合规'],
            'gold_status':'human_review_required', **extra})
    for category, questions in COMPANY.items():
        for q in questions: add('company',q,category=category, expected_sources=['company_rag'])
    for q, files, points in ATTACHMENTS:
        add('attachment',q,files=files,expected_points=points,expected_sources=['customer_documents'],
            gold_status='synthetic_fixture_defined')
    for q in WEB: add('web',q,use_online_search=True,rubric=['对象与事实日期匹配','页面日期不等于事实日期',
        '预测不等于实测','多个目标分别召回','联网无结果或额度不足应明确说明'])
    for i,(turns, expectation) in enumerate(TURNS):
        files = ['a.txt','b.txt'] if any('方案' in t for t in turns) else []
        add('multiturn',turns[-1],turns=turns,files=files,
            mode='steer' if i in {0,2,5,6,8,9} else 'queue' if i in {14,18} else 'history',
            use_online_search=any('天气' in t or '网络' in t for t in turns),
            rubric=[expectation,'最新要求优先，不把助手响应当用户任务','历史与当前任务明确区分'])
    fault_modules = ['backend.tests.sales.test_model_contract','backend.tests.sales.test_output_contract',
                     'backend.tests.test_task_cancellation','backend.tests.sales.test_recovery_budget',
                     'backend.tests.sales.test_agent_eval_100']
    for name,module in zip(['格式最多重试两次','引用未知ID明确失败','权限与取消隔离','恢复预算耗尽',
                            '总时限与超时取消'],fault_modules):
        add('fault',name,mode='fault',test_module=module,gold_status='deterministic_protocol_test')
    assert len(rows)==100 and len({r['id'] for r in rows})==100
    return rows


def build():
    DEST.mkdir(parents=True,exist_ok=True)
    fixtures=DEST/'fixtures'; fixtures.mkdir(exist_ok=True)
    (fixtures/'a.txt').write_text('方案A：A板厚度18毫米，室外干挂，不适用于室内。价格120元/平方米，含税。付款先付30%，验收后付70%。',encoding='utf-8')
    (fixtures/'b.txt').write_text('方案B：B板厚度25毫米，粘锚安装，适用于室外，不允许水下使用。价格150元/平方米，不含税。付款先付40%，验收后付60%。',encoding='utf-8')
    (fixtures/'records.csv').write_text('日期,工厂,产量,单位\n2026-10-07,甲厂,12,吨\n2026-10-08,甲厂,20,吨\n2026-10-08,乙厂,30,吨\n2026-10-09,甲厂,99,吨\n',encoding='utf-8-sig')
    (fixtures/'long.txt').write_text(('安装准备：检查基层，单板采用干挂，保持接缝。\n'*1800)+'最后禁止事项：禁止浸水。',encoding='utf-8')
    from PIL import Image,ImageDraw
    img=Image.new('RGB',(480,240),'white'); draw=ImageDraw.Draw(img)
    draw.rectangle((35,45,175,185),fill='red'); draw.ellipse((285,45,425,185),fill='blue')
    img.save(fixtures/'shapes.png')
    path=DEST/'questions.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases()),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures.iterdir() if p.is_file()}
    (DEST/'manifest.json').write_text(json.dumps({'version':1,'case_count':100,'question_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'fixture_sha256':hashes,'semantic_gold':'company/web/multiturn require independent review; lexical checks are diagnostics'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Built {len(cases())} cases: {path}')


if __name__=='__main__': build()
