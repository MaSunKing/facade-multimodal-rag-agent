import test from 'node:test';
import assert from 'node:assert/strict';
import {attachmentSnapshot} from '../app/sales/attachment-snapshot.ts';
import {attachmentKind} from '../app/sales/attachment-kind.ts';

test('display suffix never changes source filename or multimodal kind', () => {
  for (const [name,kind] of [['image.png','图片'],['file.pdf','PDF'],['file.docx','Word'],['file.xlsx','表格']]) {
    const row=attachmentSnapshot('s',[{file_name:name,display_name:name+'（2）',document_id:'d'}])[0];
    assert.equal(row.name,name);
    assert.equal(row.displayName,name+'（2）');
    assert.equal(attachmentKind(row.name),kind);
  }
});

test('associated files are recorded without a new upload and without file bytes', () => {
  const result = attachmentSnapshot('session-1', [
    {document_id:'d1',file_name:'image.png',preview_visual_id:'v1',preview:'blob:temporary'},
    {document_id:'d2',file_name:'report.pdf'},
  ]);
  assert.deepEqual(result, [{name:'image.png',sessionId:'session-1',documentId:'d1',visualId:'v1'},
    {name:'report.pdf',sessionId:'session-1',documentId:'d2',visualId:undefined}]);
  assert.ok(!JSON.stringify(result).includes('blob:'));
});

test('pending and expired file cards retain names; file count is bounded', () => {
  assert.deepEqual(attachmentSnapshot(null, [], ['report.docx']), [{name:'report.docx',sessionId:undefined}]);
  assert.equal(attachmentSnapshot('s', [], ['a','b','c','d','e']).length, 4);
  assert.deepEqual(attachmentSnapshot('s', [{file_name:' '},{}], ['fallback.xlsx']), [{name:'fallback.xlsx',sessionId:'s'}]);
});
