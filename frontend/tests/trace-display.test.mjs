import assert from 'node:assert/strict';
import test from 'node:test';
import {compactDetail, recorded, importantStage, stageTokens, normalizeTraceId, preferredTraceId} from '../app/sales/trace-display.ts';

test('running client task takes precedence over previous completed answer', () => {
  const task = '12345678-1234-1234-1234-123456789abc';
  assert.equal(preferredTraceId(task, 'req_'+'a'.repeat(32)), task);
  assert.equal(preferredTraceId('', 'req_'+'a'.repeat(32)), 'req_'+'a'.repeat(32));
});

test('empty diagnostics do not become fake nodes or placeholders', () => {
  for (const value of [null, {}, [], [{}, null], 'unknown', '[depth limit]']) assert.equal(recorded(value), false);
  assert.equal(importantStage({stage:'generate_answer',phase:'model_loading'}), false);
  assert.equal(importantStage({stage:'assess_coverage'}), false);
  assert.equal(importantStage({stage:'plan_request'}), true);
  assert.equal(importantStage({stage:'other',error_code:'FAILED'}), true);
});
test('expansion and answer audit display actual recorded token usage', () => {
  assert.equal(importantStage({stage:'evidence_expansion'}),true);
  assert.equal(importantStage({stage:'answer_audit'}),true);
  assert.equal(stageTokens({token_usage:{model_calls:0}}),'0（程序处理）');
  assert.equal(stageTokens({token_usage:{model_calls:2,input_tokens:100,output_tokens:40}}),'输入 100 / 输出 40');
  assert.equal(stageTokens({}), '未记录');
  assert.equal(normalizeTraceId('排查编号：'+'a'.repeat(32)),'a'.repeat(32));
  assert.equal(normalizeTraceId('req_'+'b'.repeat(32)),'req_'+'b'.repeat(32));
});
test('real negative audit results survive compact presentation', () => {
  assert.deepEqual(compactDetail({empty:[], absent:null, valid:false, count:0, result:{}, tools:['company_rag']}),
                   {valid:false,count:0,tools:['company_rag']});
});
