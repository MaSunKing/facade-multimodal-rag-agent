"""Scoring checks only; do not modify production retrieval/context."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_basic_capabilities_extra_eval import numeric_flags, conflict_support, cases


def test_numeric_requires_native_unit_binding():
    sample=cases()[0]
    answer=dict(document_name=sample['file'],text="[TABLE sheet=epa_01_01]\n[ROW] A12='Nuclear' | D12='774873'")
    assert numeric_flags([answer],sample)['target_row_value_retained']
    assert not numeric_flags([answer],sample)['row_value_unit_retained']
    unit=dict(document_name=sample['file'],text="[TABLE sheet=epa_01_01]\n[ROW] A6='Net Generation (Thousand Megawatthours)'")
    assert numeric_flags([answer,unit],sample)['row_value_unit_retained']
    assert not numeric_flags([answer,dict(unit,document_name='other.xlsx')],sample)['row_value_unit_retained']


def test_same_number_wrong_cell_is_not_support():
    sample=cases()[0]
    item=dict(document_name=sample['file'],text="[TABLE sheet=epa_01_01]\n[ROW] D15='774873'")
    assert not numeric_flags([item],sample)['target_row_value_retained']


def test_conflict_requires_both_files_and_complete_measurements():
    sample=cases()[10]
    items=[dict(document_name=f,text=f'{sample["entity"]} thickness: {v} mm.') for f,v in zip(sample['files'],sample['values'])]
    assert conflict_support(items,sample)
    assert not conflict_support([items[0]],sample)
    assert not conflict_support([items[0],dict(items[1],document_name=sample['files'][0])],sample)
