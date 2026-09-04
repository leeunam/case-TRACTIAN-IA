import type { CaseDetail, Decision, DemoCase, DemoConfig, Persona } from "./types";

const BASE = import.meta.env.VITE_DEMO_API_URL ?? "http://127.0.0.1:8100";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.detail?.message ?? "A operação não pôde ser concluída.");
  }
  return response.json() as Promise<T>;
}

export const api = {
  config: () => request<DemoConfig>("/v1/demo/config"),
  personas: () => request<Persona[]>("/v1/personas"),
  cases: () => request<DemoCase[]>("/v1/cases"),
  case: (id: string) => request<CaseDetail>(`/v1/cases/${id}`),
  decisions: (personaId: string) => request<Decision[]>(`/v1/decisions?persona_id=${encodeURIComponent(personaId)}`),
  duplicate: (sourceCaseId: string) => request<DemoCase>("/v1/cases", { method: "POST", body: JSON.stringify({ source_case_id: sourceCaseId }) }),
  create: (data: { company_id: string; requester_id: string; asset_id: string; message: string }) => request<DemoCase>("/v1/cases", { method: "POST", body: JSON.stringify(data) }),
  message: (caseId: string, personaId: string, content: string) => request(`/v1/cases/${caseId}/messages`, { method: "POST", body: JSON.stringify({ persona_id: personaId, content, idempotency_key: crypto.randomUUID() }) }),
  resolve: (decisionId: string, personaId: string, resolution: "approve" | "reject") => request(`/v1/decisions/${decisionId}/resolve`, { method: "POST", body: JSON.stringify({ persona_id: personaId, resolution }) }),
  eventUrl: (caseId: string) => `${BASE}/v1/cases/${caseId}/events`,
};
