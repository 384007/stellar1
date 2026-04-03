"use client";

const DB_NAME = "stellar-history-db";
const STORE_NAME = "analysis_videos";
const DETAIL_STORE = "analysis_details";
const DB_VERSION = 2;

interface StoredVideoRecord {
  analysisId: string;
  blob: Blob;
  filename: string;
  mimeType: string;
  createdAt: string;
}

interface StoredDetailRecord {
  analysisId: string;
  json: string;
  createdAt: string;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "analysisId" });
      }
      if (!db.objectStoreNames.contains(DETAIL_STORE)) {
        db.createObjectStore(DETAIL_STORE, { keyPath: "analysisId" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveAnalysisVideo(
  analysisId: string,
  blob: Blob,
  filename: string
): Promise<void> {
  if (!analysisId || !blob || blob.size === 0) return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const record: StoredVideoRecord = {
      analysisId,
      blob,
      filename,
      mimeType: blob.type || "video/webm",
      createdAt: new Date().toISOString(),
    };
    store.put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  db.close();
}

export async function getAnalysisVideoBlob(analysisId: string): Promise<Blob | null> {
  if (!analysisId) return null;
  const db = await openDb();
  const record = await new Promise<StoredVideoRecord | undefined>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(analysisId);
    req.onsuccess = () => resolve(req.result as StoredVideoRecord | undefined);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return record?.blob ?? null;
}

export async function saveAnalysisDetail(
  analysisId: string,
  json: string,
): Promise<void> {
  if (!analysisId || !json) return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(DETAIL_STORE, "readwrite");
    const store = tx.objectStore(DETAIL_STORE);
    const record: StoredDetailRecord = {
      analysisId,
      json,
      createdAt: new Date().toISOString(),
    };
    store.put(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  db.close();
}

export async function getAnalysisDetail(analysisId: string): Promise<string | null> {
  if (!analysisId) return null;
  const db = await openDb();
  const record = await new Promise<StoredDetailRecord | undefined>((resolve, reject) => {
    const tx = db.transaction(DETAIL_STORE, "readonly");
    const store = tx.objectStore(DETAIL_STORE);
    const req = store.get(analysisId);
    req.onsuccess = () => resolve(req.result as StoredDetailRecord | undefined);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return record?.json ?? null;
}

export async function deleteAnalysisVideo(analysisId: string): Promise<void> {
  if (!analysisId) return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    store.delete(analysisId);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  db.close();
}

export async function deleteAnalysisDetail(analysisId: string): Promise<void> {
  if (!analysisId) return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(DETAIL_STORE, "readwrite");
    const store = tx.objectStore(DETAIL_STORE);
    store.delete(analysisId);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
  db.close();
}
