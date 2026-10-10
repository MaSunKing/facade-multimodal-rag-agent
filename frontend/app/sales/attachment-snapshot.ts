export type MessageAttachment = {name: string; displayName?: string; sessionId?: string; documentId?: string; visualId?: string};
export type SessionDocument = {file_name?: string; display_name?: string; document_id?: string; preview_visual_id?: string};

// Store source references, never customer file bytes or temporary blob URLs.
export function attachmentSnapshot(sessionId: string | null, documents: SessionDocument[], fallbackNames: string[] = []): MessageAttachment[] {
  const rows = documents.filter(row => row.file_name?.trim()).slice(0, 4);
  return rows.length ? rows.map(row => ({name: row.file_name!.trim(), ...(row.display_name && row.display_name !== row.file_name ? {displayName:row.display_name} : {}),
    sessionId: sessionId ?? undefined, documentId: row.document_id, visualId: row.preview_visual_id}))
    : fallbackNames.slice(0, 4).map(name => ({name, sessionId: sessionId ?? undefined}));
}
