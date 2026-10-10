"use client";
import {useEffect, useRef, useState} from "react";
import {FileBadge} from "./session-attachments";
import {attachmentKind} from "./attachment-kind";
import type {MessageAttachment} from "./attachment-snapshot";

export default function MessageAttachments({items, baseUrl, authenticatedFetch}: {
  items: MessageAttachment[]; baseUrl: string;
  authenticatedFetch: (url: string, init?: RequestInit) => Promise<Response>;
}) {
  const fetchRef = useRef(authenticatedFetch); fetchRef.current = authenticatedFetch;
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const signature = JSON.stringify(items);
  useEffect(() => {
    const controller = new AbortController();
    const urls: string[] = [];
    setPreviews({});
    void Promise.all((JSON.parse(signature) as MessageAttachment[]).map(async item => {
      // A Word/PDF embedded illustration is not a preview of the file itself.
      if (attachmentKind(item.name) !== "图片" || !item.sessionId || !item.documentId || !item.visualId) return;
      try {
        const response = await fetchRef.current(`${baseUrl}/api/copilot/documents/${encodeURIComponent(item.sessionId)}/visual/${encodeURIComponent(item.documentId)}/${encodeURIComponent(item.visualId)}`, {signal: controller.signal});
        if (!response.ok || !response.headers.get("content-type")?.startsWith("image/")) return;
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        const url = URL.createObjectURL(blob); urls.push(url);
        setPreviews(current => ({...current, [item.documentId!]: url}));
      } catch { /* Expired sessions keep their filename/type card. */ }
    }));
    return () => { controller.abort(); urls.forEach(url => URL.revokeObjectURL(url)); };
  }, [signature, baseUrl]);
  return <div className="attachment-card-list message-attachments" aria-label="本次关联附件">
    {items.map((item, index) => <div className="attachment-card" key={`${item.documentId ?? item.name}-${index}`}>
      <FileBadge name={item.name} preview={item.documentId ? previews[item.documentId] : undefined}/>
      <span title={item.name}>{item.displayName || item.name}<small>{attachmentKind(item.name)} · 本次关联</small></span>
    </div>)}
  </div>;
}
