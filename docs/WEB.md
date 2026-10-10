# Web MCP、排序、缓存与时效

[首页](../README.md) → [轨迹](TRACES.md)

[search_mcp.py](../backend/sales/search_mcp.py)通过真实MCP初始化与tools/call调用network_search；当前后端使用进程内MCP transport以保留请求预算/取消ContextVar，也可stdio服务。MCP只规范工具接口，不会提高搜索引擎本身的召回质量，也不是额外独立联网Agent。

```text
Planner选择public_web_search + 完整独立queries
 → 后端授权、公开查询及当前日期/范围
 → web_requirements逐要求调用network_search
 → 千帆Web API默认请求20条（上游不保证每次正好20条）
 → 分数完整有效时按rerank_score降序稳定排序
   缺失/无效时保留接口原顺序，不补造分数
 → 来源质量/重复/范围筛选 → 最多5条对外工具来源
 → 配置允许时核验少量正文（默认前2条，上限3）
 → 附Requirement归属与事实日期 → Web Context
```

实际实现：[baidu_search.py](../backend/sales/baidu_search.py)、[web_quality.py](../backend/sales/web_quality.py)、[web_requirements.py](../backend/sales/web_requirements.py)、[web_context.py](../backend/sales/web_context.py)。质量筛选不保证消除所有垃圾或所有不相关内容。

多对象任务不能把“新加坡今天的天气＋温州呢”拼成唯一Query；有效意图统一、检索要求分别执行、证据保留归属、最后一次综合回答。补取证按剩余额度，不无界增加联网调用。

缓存节省调用，但旧来源的事实时间不能变成今天；bypass_cache用于明确重新检索，不等于刷新旧网页。缓存命中也要按当前实体、日期、版本和条件过滤。发布日期、抓取时间和正文事实日期分别看，实时题不能仅凭网页标题“今天”通过。

默认网络搜索业务额度与硬额度分别配置，耗尽返回明确状态而不绕过限制。只提交授权公开查询，不把客户文件、企业私有Evidence或聊天历史发给API；当前问题本身可能含敏感信息，授权不代替隐私审查。正文核验遵守受控网络访问、大小/超时与来源边界；摘要仍标正文未核验时不能升级为认证事实。
