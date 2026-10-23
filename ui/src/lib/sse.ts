import { API_URL, ApiError, KEY_HEADER, errorFromResponse } from "./api";
import type { LiveEvent } from "./types";

export interface RawEvent {
  event: string;
  data: unknown;
}

/**
 * Split a buffer of server-sent-event text into complete events plus the unfinished remainder.
 * Handles \r\n and \n line endings and multi-line data fields.
 */
export function parseSse(buffer: string): { events: RawEvent[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const blocks = normalized.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: RawEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // comment / keep-alive ping
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) continue;
    try {
      events.push({ event, data: JSON.parse(data.join("\n")) });
    } catch {
      /* ignore a malformed event rather than abort the whole stream */
    }
  }
  return { events, rest };
}

export interface RunRequest {
  question?: string;
  question_id?: string;
  score: boolean;
}

/**
 * Start a live run and stream its events. The key goes in a request header, never in the URL or
 * body, and is not retained anywhere by this function.
 */
export async function streamLiveRun(
  key: string,
  body: RunRequest,
  onEvent: (e: LiveEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/run`, {
      method: "POST",
      headers: { [KEY_HEADER]: key, "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Network error", 0, "network");
  }
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError("The server sent no stream", 502, "upstream_error");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSse(buffer);
    buffer = parsed.rest;
    for (const ev of parsed.events) onEvent({ event: ev.event, data: ev.data } as LiveEvent);
  }
  buffer += decoder.decode();
  for (const ev of parseSse(buffer + "\n\n").events) {
    onEvent({ event: ev.event, data: ev.data } as LiveEvent);
  }
}
