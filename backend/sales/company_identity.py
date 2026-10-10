"""Configured service identity, not a keyword router or product fact generator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IDENTITY_PATH = Path(__file__).resolve().parents[2] / "data/sales/config/company_identity.json"


def load_company_identity() -> dict[str, Any]:
    if not IDENTITY_PATH.is_file():
        return {}
    return json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))


def identity_prompt_context() -> str:
    identity = load_company_identity()
    if not identity:
        return ""
    return (
        "\n服务身份（配置，不是权限或路由规则）："
        + str(identity["legal_name"])
        + "；品牌：" + ", ".join(identity["brand_names"])
        + "。询问本服务企业时，you/your company/你们/我们公司 指配置的厂家，不是模型提供商。"
          "完整检索或联网查询中的明确公司指代可消解为该公司全名。"
          "不替换用户明确提到的其他实体，不改变无关话题。公司与产品事实仍需引用本地或官方证据，"
          "不得编造额外产品线。\n"
    )
