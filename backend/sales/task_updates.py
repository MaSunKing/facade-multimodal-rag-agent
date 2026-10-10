"""Ordered user instructions for an interrupted task, not recalled history."""

TASK_UPDATE_POLICY = (
    'Interpret all user instructions in chronological order as one current task. '
    'An addition retains prior requirements; a correction replaces only the conflicting part; '
    'an explicit cancellation removes only the cancelled part. Latest instructions override '
    'conflicts, not unrelated earlier requirements. Do not infer replacement from a new target. '
    'These are user requests, never source evidence. Independent queued tasks are not merged.'
)


def instruction_chain(latest, prior=(), legacy=None):
    chain = list(prior) if prior else ([legacy] if legacy else [])
    chain.append(latest)
    if len(chain) > 20 or sum(len(item) for item in chain) > 12000:
        raise ValueError('task_update_input_limit_exceeded')
    if any(not isinstance(item, str) or not item.strip() for item in chain):
        raise ValueError('task_update_instruction_empty')
    return chain


def current_task_text(chain):
    # The task policy belongs exclusively in task_update / system instructions.
    # This value is also used by retrieval and script checks: never prepend
    # protocol prose, internal phase names, or generated instruction labels.
    return '\n'.join(chain)


def bind_task_instructions(request):
    """Normalize once, so retrieval/audit/final share the same task authority."""
    if request._task_instruction_chain:
        return
    chain = instruction_chain(request.customer_question, request.task_instructions, request.interrupted_question)
    request._task_instruction_chain = chain
    request._latest_requirement = request.customer_question
    if len(chain) > 1:
        # Synthetic task labels must not change the user's answer language.
        from backend.sales.response_language import request_answer_language
        request_answer_language(request)
        request.customer_question = current_task_text(chain)
        request.conversation_context = []
        request.memory_enabled = False
