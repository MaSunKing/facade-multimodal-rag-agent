"""Loopback-only Qdrant adapter. SQL/canonical index remains authoritative."""
import json
import os
import uuid
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


def point_id(evidence_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, 'facade-evidence:'+evidence_id))


def collection_name(fingerprint):
    if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('vector_fingerprint_invalid')
    return 'facade_'+fingerprint


class QdrantStore:
    def __init__(self, url=None):
        self.url = (url or os.getenv('RAG_QDRANT_URL', 'http://127.0.0.1:6333')).rstrip('/')
        parsed = urlsplit(self.url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1','localhost','::1'} or parsed.username or parsed.password or parsed.path:
            raise ValueError('vector_service_must_be_loopback')

    def call(self, method, path, data=None):
        raw = None if data is None else json.dumps(data, allow_nan=False).encode()
        request = Request(self.url+path, data=raw, method=method, headers={'Content-Type':'application/json'})
        # Transport timeout only: not an Agent request/stage deadline.
        with urlopen(request, timeout=30) as response:
            return json.load(response).get('result')

    def ready(self, fingerprint, dimension, count):
        value = self.call('GET', '/collections/'+collection_name(fingerprint))
        params = value['config']['params']['vectors']
        return params['size'] == dimension and params['distance'] == 'Cosine' and value['points_count'] == count

    def search(self, vector, *, fingerprint, allowed_ids, limit=48):
        # Always prefilter by current canonical visibility AND relation scope.
        # Empty scope is empty, never a request to search all private points.
        allowed = set(allowed_ids)
        if not allowed:
            return []
        data = {'query': [float(x) for x in vector], 'limit':limit, 'with_payload':True,
                'filter': {'must':[{'has_id':[point_id(x) for x in sorted(allowed)]}]}}
        result = self.call('POST','/collections/'+collection_name(fingerprint)+'/points/query',data)
        rows = result['points']
        # Treat service output as untrusted: exact IDs/fingerprint rechecked.
        return [(p['payload']['evidence_id'], float(p['score'])) for p in rows
                if p.get('payload',{}).get('evidence_id') in allowed
                and p.get('id') == point_id(p['payload']['evidence_id'])
                and p['payload'].get('fingerprint') == fingerprint]

    def import_index(self, ids, vectors, fingerprint):
        name = collection_name(fingerprint)
        try:
            if self.ready(fingerprint, vectors.shape[1], len(ids)):
                return {'collection':name,'count':len(ids),'reused':True}
        except Exception:
            pass
        # A fingerprint-named collection is versioned; never delete old data.
        from urllib.error import HTTPError
        try:
            self.call('GET','/collections/'+name)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            self.call('PUT','/collections/'+name,{'vectors':{'size':vectors.shape[1],'distance':'Cosine'}})
        for start in range(0,len(ids),128):
            self.call('PUT','/collections/'+name+'/points?wait=true',{'points':[
                {'id':point_id(key),'vector':[float(v) for v in vector],
                 'payload':{'evidence_id':key,'fingerprint':fingerprint,'origin':'company'}}
                for key,vector in zip(ids[start:start+128], vectors[start:start+128])]})
        if not self.ready(fingerprint, vectors.shape[1], len(ids)):
            raise ValueError('vector_import_verification_failed')
        return {'collection':name,'count':len(ids),'reused':False}
