"""Request language policy, independent of evidence language and tool routing."""
import re
from backend.document_parsing.ingestion import language_candidates_for_text

NAMES = {'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
         'ru': 'Russian', 'ar': 'Arabic'}


def resolve_answer_language(question):
    # Only explicit language preferences; never infer a tool or task from these.
    english = re.findall(r'\b(?:answer|respond|reply|write)\s+(?:to me\s+)?in\s+([A-Za-z]+)', question, re.I)
    if english:
        aliases = {'english': 'English', 'chinese': 'Chinese', 'mandarin': 'Chinese',
                   'spanish': 'Spanish', 'french': 'French', 'german': 'German',
                   'japanese': 'Japanese', 'korean': 'Korean', 'russian': 'Russian', 'arabic': 'Arabic'}
        if english[-1].lower() in aliases:
            return aliases[english[-1].lower()]
    names = {'中文':'Chinese', '汉语':'Chinese', '英文':'English', '英语':'English',
             '日语':'Japanese', '日文':'Japanese', '韩语':'Korean', '法语':'French',
             '德语':'German', '西班牙语':'Spanish', '俄语':'Russian', '阿拉伯语':'Arabic'}
    matches = re.findall(r'(?:请用|使用|用|以)\s*(' + '|'.join(names) + r')\s*(?:回答|回复|作答|解释|写)', question)
    if matches:
        return names[matches[-1]]
    hints = language_candidates_for_text(question)
    if hints and hints[0].script == 'Latn' and re.search(
            r'\b(?:hello|how|what|why|where|when|which|who|please|today|weather|are|you|your|does|can|could|would)\b',
            question, re.I):
        return 'English'
    if hints and hints[0].code in NAMES:
        return NAMES[hints[0].code]
    return '遵循当前用户明确要求，否则使用当前问题语言（不跟随证据语言）'


def request_answer_language(request):
    if not request._answer_language:
        request._answer_language = resolve_answer_language(request.customer_question)
    return request._answer_language


def compatible_web_query(question, query):
    """Reject obvious script switches; this does not prove translation semantics."""
    query = (query or '').strip()
    if not query:
        return question.strip()
    def primary_script(text):
        hints = language_candidates_for_text(text)
        return hints[0].script if hints else None
    original_script, rewritten_script = primary_script(question), primary_script(query)
    if original_script and rewritten_script and original_script != rewritten_script:
        return question.strip()
    return query


def web_caution(language):
    return {'English': 'The following information comes from public web leads that have not been verified against official sources; use it as a reference only:',
            'Chinese': '以下内容来自尚未完成官网核验的公开网页线索，仅供参考：'}.get(language, '')


def evidence_gate_notice(language, *, audit_failed=False, ambiguous=False):
    """Localised program response, not an extra model generation."""
    if language == 'English':
        if audit_failed:
            return ('The evidence review could not be completed. This does not mean the information is unavailable.',
                    'Please retry the current question.', 'Evidence relevant to the current question')
        return ('The selected evidence does not reliably support a complete answer to this question.',
                'Please clarify the intended subject.' if ambiguous else 'Please provide relevant information or try again later.',
                'Evidence relevant to the current question')
    if audit_failed:
        return ('本次审核未完成，尚不能判断现有资料是否足够支持回答。', '请重试当前问题。', '当前问题的相关证据')
    return ('本次选中资料尚不能可靠支持完整回答。',
            '请明确所指对象。' if ambiguous else '可补充相关资料或稍后重试。', '当前问题的相关证据')
