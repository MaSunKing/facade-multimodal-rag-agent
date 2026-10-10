"""Generation-token accounting from actual model attempts, never text estimates."""


def usage(records):
    attempts = [a for r in records for a in r.get('attempts', [])]
    unknown = sum(a.get('output_tokens') is None for a in attempts)
    return {'input_tokens': sum(a.get('input_tokens', 0) for a in attempts),
            'output_tokens': sum(a.get('output_tokens', 0) or 0 for a in attempts),
            'total_tokens': sum((a.get('input_tokens', 0) or 0) + (a.get('output_tokens', 0) or 0) for a in attempts),
            'model_calls': len(attempts), 'unknown_output_calls': unknown,
            'complete': unknown == 0,
            'scope': 'generation_only_inclusive_children',
            'policy': 'actual_attempts_including_format_retries; do_not_sum_parent_and_child'}
