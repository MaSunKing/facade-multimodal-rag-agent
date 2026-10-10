import unittest

from backend.sales.answer_patch import diagnose, normalize_checklist, build_patch_payload, merge_patch


def fixture():
    evidence = {k: {'evidence_id': k, 'document_id': d, 'text': text}
                for k, d, text in [('U1', 'A', 'A is a resume.'), ('U2', 'B', 'B is a budget.') ]}
    aspects = ['describe A', 'describe B']
    bindings = [{'document_id': d, 'document_name': d, 'aspect_index': i} for i, d in enumerate(['A', 'B'])]
    result = {'customer_reply': 'A is a resume.', 'citations': [{'evidence_id': 'U1'}],
              'answer_aspect_coverage': [{'aspect_index': 0, 'status': 'answered',
                                        'answer_quote': 'A is a resume.', 'evidence_ids': ['U1']}]}
    return result, aspects, bindings, evidence


def payload_for(result, aspects, bindings, evidence):
    return build_patch_payload(result=result, question='Describe both files', language='en',
        aspects=aspects, bindings=bindings, evidence=evidence,
        verdict={'status': 'supported', 'supported_evidence_ids': ['U1', 'U2']})


def test_exact_gap_and_only_missing_file_evidence():
    result, aspects, bindings, evidence = fixture()
    gap = diagnose(result, aspects, bindings, evidence)
    assert gap['missing_documents'] == ['B']
    assert gap['failures'] == [{'aspect_index': 1, 'reasons': ['checklist_entry_missing']}]
    payload, deferred = payload_for(result, aspects, bindings, evidence)
    assert not deferred
    assert [r['evidence_id'] for r in payload['evidence']] == ['U2']
    assert payload['answer_language'] == 'en'


def test_syntax_repair_does_not_infer_support():
    result, aspects, bindings, evidence = fixture()
    result['answer_aspect_coverage'][0].update(aspect_index='0', evidence_ids='U1')
    normalized, changes = normalize_checklist(result, evidence)
    assert len(changes) == 2
    assert diagnose(normalized, aspects, bindings, evidence)['missing_aspects'] == [1]
    assert result['answer_aspect_coverage'][0]['aspect_index'] == '0'


def test_missing_audit_support_deferred():
    result, aspects, bindings, evidence = fixture()
    payload, deferred = build_patch_payload(result=result, question='both', language='zh',
        aspects=aspects, bindings=bindings, evidence=evidence,
        verdict={'status': 'partial', 'supported_evidence_ids': ['U1']})
    assert payload['targets'] == []
    assert deferred[0]['aspect_index'] == 1


def test_patch_append_preserves_answer_and_coverage():
    result, aspects, bindings, evidence = fixture()
    payload, _ = payload_for(result, aspects, bindings, evidence)
    merged, audit = merge_patch(result, {'additions': [{'aspect_index': 1,
        'status': 'answered', 'text': 'B is a budget.', 'evidence_ids': ['U2']}]},
        payload=payload, aspects=aspects, bindings=bindings, evidence=evidence)
    assert merged['customer_reply'] == 'A is a resume.\n\nB is a budget.'
    assert audit['missing_documents'] == audit['missing_aspects'] == []
    assert result['customer_reply'] == 'A is a resume.'


def test_reject_invalid_patch():
    for field, value in [('evidence_ids', ['U1']), ('evidence_ids', ['unknown']),
                         ('aspect_index', 0), ('aspect_index', True), ('text', ''), ('status', 'complete')]:
        result, aspects, bindings, evidence = fixture()
        payload, _ = payload_for(result, aspects, bindings, evidence)
        row = {'aspect_index': 1, 'status': 'answered', 'text': 'B is a budget.', 'evidence_ids': ['U2']}
        row[field] = value
        with unittest.TestCase().assertRaises(ValueError):
            merge_patch(result, {'additions': [row]}, payload=payload,
                        aspects=aspects, bindings=bindings, evidence=evidence)


def test_visual_support_not_patched_as_text():
    result, aspects, bindings, evidence = fixture()
    payload, deferred = build_patch_payload(result=result, question='both', language='zh',
        aspects=aspects, bindings=bindings, evidence=evidence,
        verdict={'status': 'supported', 'supported_evidence_ids': ['U1', 'U2'], 'visual_need': 'required'})
    assert not payload['targets'] and deferred


def test_explicit_missing_is_not_supported_summary():
    result, aspects, bindings, evidence = fixture()
    payload, _ = payload_for(result, aspects, bindings, evidence)
    merged, audit = merge_patch(result, {'additions': [{'aspect_index': 1,
        'status': 'evidence_missing', 'text': 'B has insufficient evidence for this question.', 'evidence_ids': []}]},
        payload=payload, aspects=aspects, bindings=bindings, evidence=evidence)
    assert audit['aspect_audit']['covered_aspect_indices'] == [0]
    assert audit['aspect_audit']['evidence_missing_aspect_indices'] == [1]


def test_file_check_rejects_wrong_document_even_if_aspect_passes():
    result, aspects, bindings, evidence = fixture()
    result['customer_reply'] += ' B is discussed.'
    result['answer_aspect_coverage'].append({'aspect_index': 1, 'status': 'answered',
        'answer_quote': 'B is discussed.', 'evidence_ids': ['U1']})
    gaps = diagnose(result, aspects, bindings, evidence)
    assert not gaps['missing_aspects'] and gaps['missing_documents'] == ['B']


def test_unknown_citation_is_observable():
    result, aspects, bindings, evidence = fixture()
    result['citations'].append({'evidence_id': 'invented'})
    assert diagnose(result, aspects, bindings, evidence)['citation_errors'] == [1]


def test_malformed_citation_diagnostic_does_not_crash():
    result, aspects, bindings, evidence = fixture()
    result['citations'].append({'evidence_id': ['bad']})
    assert diagnose(result, aspects, bindings, evidence)['citation_errors'] == [1]


def test_admin_projection_never_exposes_answer_or_requirements():
    from backend.sales.admin_traces import correction_view
    projected = correction_view({'initial': {'missing_aspects': [1],
        'aspect_audit': {'requested_aspects': ['PRIVATE_CONTENT']},
        'file_audit': {'documents': [{'document_name': 'PRIVATE_CONTENT'}]}}})
    assert projected['initial']['missing_aspects'] == [1]
    assert 'PRIVATE_CONTENT' not in str(projected)


def grounding_for(result, patch, merged, payload, evidence):
    from backend.sales.answer_patch import check_patch_grounding
    from backend.app import evidence_support_audit
    from backend.sales.answer_coverage import guard_normative_claims
    return check_patch_grounding(result, patch, merged, payload=payload, evidence=evidence,
        support_check=evidence_support_audit, normative_check=guard_normative_claims)


def test_original_numeric_problem_does_not_reject_valid_patch():
    result, aspects, bindings, evidence = fixture()
    result['customer_reply'] += ' Old unsupported revenue: 999.'
    payload, _ = payload_for(result, aspects, bindings, evidence)
    patch = {'additions': [{'aspect_index': 1, 'status': 'answered',
                          'text': 'B is a budget.', 'evidence_ids': ['U2']}]}
    merged, _ = merge_patch(result, patch, payload=payload, aspects=aspects, bindings=bindings, evidence=evidence)
    check = grounding_for(result, patch, merged, payload, evidence)
    assert check['passed']
    assert check['original']['unsupported_numeric_claims'] == ['999']
    assert check['merged']['unsupported_numeric_claims'] == ['999']
    assert check['final_validation_still_required']


def test_patch_number_cannot_borrow_support_from_another_file():
    result, aspects, bindings, evidence = fixture()
    evidence['U1']['text'] += ' Year: 2025.'
    payload, _ = payload_for(result, aspects, bindings, evidence)
    patch = {'additions': [{'aspect_index': 1, 'status': 'answered',
                          'text': 'B budget year is 2025.', 'evidence_ids': ['U2']}]}
    merged, _ = merge_patch(result, patch, payload=payload, aspects=aspects, bindings=bindings, evidence=evidence)
    check = grounding_for(result, patch, merged, payload, evidence)
    assert not check['passed']
    assert check['additions'][0]['checks']['unsupported_numeric_claims'] == ['2025']
    assert check['failures'][0]['code'] == 'unsupported_numeric_claims'


def test_normative_failure_recorded_separately():
    result, aspects, bindings, evidence = fixture()
    payload, _ = payload_for(result, aspects, bindings, evidence)
    patch = {'additions': [{'aspect_index': 1, 'status': 'answered',
                          'text': 'B budget complies with ISO 9001.', 'evidence_ids': ['U2']}]}
    merged, _ = merge_patch(result, patch, payload=payload, aspects=aspects, bindings=bindings, evidence=evidence)
    check = grounding_for(result, patch, merged, payload, evidence)
    assert not check['passed']
    assert check['additions'][0]['checks']['unsupported_normative_claims']


def test_grounding_admin_view_retains_codes_but_not_customer_content():
    from backend.sales.admin_traces import correction_view
    projected = correction_view({'grounding_checks': {'passed': False,
        'original': {'unsupported_numeric_claims': ['PRIVATE_CONTENT']},
        'additions': [{'aspect_index': 1, 'failure_codes': ['unsupported_numeric_claims'],
                      'checks': {'unsupported_numeric_claims': ['PRIVATE_CONTENT']}}],
        'failures': [{'aspect_index': 1, 'code': 'unsupported_numeric_claims'}]}})
    assert 'PRIVATE_CONTENT' not in str(projected)
    assert projected['grounding_checks']['original']['unsupported_numeric_claims_count'] == 1
    assert projected['grounding_checks']['failures'][0]['code'] == 'unsupported_numeric_claims'


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))


def test_checklist_gap_does_not_trigger_content_patch():
    from backend.sales.answer_patch import requires_content_patch
    result, aspects, bindings, evidence = fixture()
    assert not requires_content_patch(diagnose(result, aspects, bindings, evidence))


def test_document_input_coverage_without_answer_checklist():
    from backend.sales.answer_patch import audit_document_input
    result, aspects, bindings, evidence = fixture()
    assert audit_document_input(bindings, evidence)['complete']
    evidence['U2']['text'] = ' '
    assert audit_document_input(bindings, evidence)['missing_document_ids'] == ['B']
    evidence['U2'].update(text='Navigation title', evidence_scope='navigation')
    assert audit_document_input(bindings, evidence)['missing_document_ids'] == ['B']


def test_patch_output_limits():
    from backend.sales.answer_patch import patch_output_limit
    assert [patch_output_limit(n) for n in (1, 2, 3, 4)] == [384, 640, 1024, 1024]
