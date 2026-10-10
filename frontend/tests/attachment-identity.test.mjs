import test from 'node:test';
import assert from 'node:assert/strict';
import {contentHash, attachmentDisplayNames} from '../app/sales/attachment-identity.ts';

test('all modalities use exact bytes, never names, dates or MIME', async () => {
  const a=new File(['same bytes'], 'a.png', {type:'image/png'});
  const b=new File(['same bytes'], 'renamed.pdf', {type:'application/pdf'});
  const c=new File(['other bytes'], 'a.png', {type:'image/png'});
  assert.equal(await contentHash(a),await contentHash(b));
  assert.notEqual(await contentHash(a),await contentHash(c));
});
test('same names keep distinct display ordinals without changing original names', () => {
  assert.deepEqual(attachmentDisplayNames(['image.png','report.pdf','image.png']),
    ['image.png（1）','report.pdf','image.png（2）']);
});
