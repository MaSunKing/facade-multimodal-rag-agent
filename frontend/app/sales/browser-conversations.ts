"use client";

/**
 * Browser-only conversation persistence.
 *
 * This module deliberately has no HTTP calls. IndexedDB is scoped to the
 * visitor's current browser profile and origin, so a conversation is not
 * copied to the local model backend, Tencent Cloud, or the RAG index.
 */

export type BrowserConversation<Message> = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
  // Only an opaque, temporary local-backend handle is persisted. File bytes
  // remain in backend process memory and are never written to browser history.
  documentSessionId?: string | null;
  attachmentNames?: string[];
};

const DATABASE_NAME = "facade-copilot-browser-history";
const DATABASE_VERSION = 1;
const STORE_NAME = "conversations";
const FALLBACK_STORAGE_KEY = "facade-copilot-browser-history-fallback-v1";
const MAX_CONVERSATIONS = 24;

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        const store = database.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("updatedAt", "updatedAt");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB unavailable"));
  });
}

function sortNewestFirst<Message>(items: BrowserConversation<Message>[]): BrowserConversation<Message>[] {
  return [...items].sort((left, right) => right.updatedAt - left.updatedAt);
}

function readFallback<Message>(): BrowserConversation<Message>[] {
  try {
    const raw = window.localStorage.getItem(FALLBACK_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? sortNewestFirst(parsed) : [];
  } catch {
    return [];
  }
}

function writeFallback<Message>(items: BrowserConversation<Message>[]) {
  try {
    window.localStorage.setItem(FALLBACK_STORAGE_KEY, JSON.stringify(sortNewestFirst(items).slice(0, MAX_CONVERSATIONS)));
  } catch {
    // Storage can be disabled in a private browser profile. The chat remains
    // usable for the current page session even when it cannot be persisted.
  }
}

export async function loadBrowserConversations<Message>(): Promise<BrowserConversation<Message>[]> {
  try {
    const database = await openDatabase();
    const items = await new Promise<BrowserConversation<Message>[]>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve(request.result as BrowserConversation<Message>[]);
      request.onerror = () => reject(request.error ?? new Error("Unable to read local conversations"));
    });
    database.close();
    return sortNewestFirst(items).slice(0, MAX_CONVERSATIONS);
  } catch {
    return readFallback<Message>();
  }
}

export async function saveBrowserConversation<Message>(conversation: BrowserConversation<Message>): Promise<void> {
  try {
    const database = await openDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(conversation);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Unable to save local conversation"));
    });

    const saved = await new Promise<BrowserConversation<Message>[]>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).getAll();
      request.onsuccess = () => resolve(request.result as BrowserConversation<Message>[]);
      request.onerror = () => reject(request.error ?? new Error("Unable to trim local conversations"));
    });
    const stale = sortNewestFirst(saved).slice(MAX_CONVERSATIONS);
    if (stale.length > 0) {
      await new Promise<void>((resolve, reject) => {
        const transaction = database.transaction(STORE_NAME, "readwrite");
        const store = transaction.objectStore(STORE_NAME);
        stale.forEach((item) => store.delete(item.id));
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error ?? new Error("Unable to trim local conversations"));
      });
    }
    database.close();
  } catch {
    const existing = readFallback<Message>().filter((item) => item.id !== conversation.id);
    writeFallback([conversation, ...existing]);
  }
}

export async function deleteBrowserConversation(conversationId: string): Promise<void> {
  try {
    const database = await openDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).delete(conversationId);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Unable to delete local conversation"));
    });
    database.close();
  } catch {
    writeFallback(readFallback<unknown>().filter((item) => item.id !== conversationId));
  }
}

export async function clearBrowserConversations(): Promise<void> {
  try {
    const database = await openDatabase();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).clear();
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Unable to clear local conversations"));
    });
    database.close();
  } catch {
    try {
      window.localStorage.removeItem(FALLBACK_STORAGE_KEY);
    } catch {
      // No recovery action is needed when browser storage is unavailable.
    }
  }
}
