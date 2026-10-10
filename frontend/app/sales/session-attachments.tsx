"use client";
import {useEffect, useRef, useState} from 'react';
import {attachmentKind} from './attachment-kind';

export function FileBadge({name, preview}: {name:string; preview?:string}) {
  return preview ? <img src={preview} alt={`${name}预览`} /> :
    <span className="attachment-file-icon" data-kind={attachmentKind(name)} aria-label={`${attachmentKind(name)}文件`}>{attachmentKind(name)}</span>;
}
type Document = {document_id:string; file_name:string; display_name?:string; preview_visual_id?:string; preview?:string};
type Props = {sessionId:string; baseUrl:string; authenticatedFetch:(url:string,init?:RequestInit)=>Promise<Response>; onExpired:()=>void; onDetach:()=>void};
export default function SessionAttachments(props: Props) {
  const callbacks = useRef(props); callbacks.current = props;
  const [documents,setDocuments] = useState<Document[]>([]);
  const [status,setStatus] = useState('checking');
  useEffect(() => {
    const controller = new AbortController();
    const urls:string[] = [];
    let checking = false;
    async function check() {
      if (checking || controller.signal.aborted) return;
      checking = true;
      try {
        const response = await callbacks.current.authenticatedFetch(`${props.baseUrl}/api/copilot/documents/${encodeURIComponent(props.sessionId)}`, {cache:'no-store',signal:controller.signal});
        if (controller.signal.aborted) return;
        if (response.status === 404) { callbacks.current.onExpired(); return; }
        if (!response.ok) throw new Error('unavailable');
        const body = await response.json();
        const rows:Document[] = (body.documents || []).slice(0,4);
        const previews = await Promise.all(rows.map(async row => {
          if (!row.preview_visual_id) return row;
          try {
            const image = await callbacks.current.authenticatedFetch(`${props.baseUrl}/api/copilot/documents/${encodeURIComponent(props.sessionId)}/visual/${encodeURIComponent(row.document_id)}/${encodeURIComponent(row.preview_visual_id)}`, {signal:controller.signal});
            if (!image.ok || !image.headers.get('content-type')?.startsWith('image/')) return row;
            const blob = await image.blob();
            if (controller.signal.aborted) return row;
            const preview = URL.createObjectURL(blob); urls.push(preview);
            return {...row,preview};
          } catch { return row; }
        }));
        if (!controller.signal.aborted) {
          setDocuments(previews); setStatus('ready');
          const currentUrls = new Set(previews.map(row => row.preview));
          for (const url of urls) if (!currentUrls.has(url)) URL.revokeObjectURL(url);
        }
      } catch { if (!controller.signal.aborted) setStatus('unavailable'); }
      finally { checking = false; }
    }
    setDocuments([]); setStatus('checking'); void check();
    window.addEventListener('focus',check);
    return () => { controller.abort(); window.removeEventListener('focus',check); urls.forEach(url=>URL.revokeObjectURL(url)); };
  }, [props.sessionId,props.baseUrl]);
  return <div className="session-attachments" aria-label="当前对话附件">
    <div className="session-attachment-heading"><span>{status === 'ready' ? '本对话可用附件' : status === 'checking' ? '正在检查历史附件…' : '暂时无法验证附件状态'}</span>
      <button type="button" onClick={()=>callbacks.current.onDetach()}>取消关联</button></div>
    {status === 'ready' && <div className="attachment-card-list">{documents.map(row => <div className="attachment-card" key={row.document_id}>
      <FileBadge name={row.file_name} preview={row.preview}/><span title={row.file_name}>{row.display_name || row.file_name}<small>{attachmentKind(row.file_name)} · 已解析</small></span>
    </div>)}</div>}
  </div>;
}
