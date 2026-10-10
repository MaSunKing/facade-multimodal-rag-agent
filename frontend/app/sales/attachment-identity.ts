// Exact byte identity only: visually similar images and same names are distinct.
export async function contentHash(file: Blob): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

export function attachmentDisplayNames(names: string[]): string[] {
  const seen = new Map<string, number>();
  return names.map(name => {
    const ordinal = (seen.get(name) ?? 0) + 1;
    seen.set(name, ordinal);
    return names.filter(value => value === name).length > 1 ? `${name}（${ordinal}）` : name;
  });
}
