import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../src/App";

const config = {
  mode: "live", warning: "Demonstração com dados e identidades simulados.",
  industrial_api: "configured", primary_provider: "groq",
  fallback_provider: "nvidia-nim", slack_configured: true,
};
const personas = [
  { id: "usr_1", name: "Joana", profile: "requester", company_id: "comp_1", permissions: ["read"] },
  { id: "boss_1", name: "Marina", profile: "authority", company_id: "comp_1", permissions: ["read", "action_high"] },
  { id: "tractian_reviewer", name: "Equipe TRACTIAN", profile: "tractian", company_id: null, permissions: ["technical_review"] },
];
const publicCase = {
  id: "case_public_1", ticket_id: "TKT-1", company_id: "comp_1",
  requester_id: "usr_1", asset_id: "asset_1", initial_message: "Ajude a analisar",
  source_case_id: null, immutable: true, created_at: "2026-01-01T00:00:00Z",
};
const detail = { case: publicCase, messages: [], executions: [] };

function mockApi() {
  const json = (value: unknown) => Promise.resolve({
    ok: true,
    json: async () => value,
  } as Response);
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    if (url.endsWith("/v1/demo/config")) return json(config);
    if (url.endsWith("/v1/personas")) return json(personas);
    if (url.endsWith("/v1/cases") && (!init || init.method !== "POST")) return json([publicCase]);
    if (url.endsWith("/v1/cases/case_public_1")) return json(detail);
    if (url.endsWith("/v1/decisions?persona_id=usr_1")) return json([]);
    if (url.endsWith("/v1/decisions?persona_id=boss_1")) return json([]);
    if (url.endsWith("/v1/decisions?persona_id=tractian_reviewer")) return json([]);
    throw new Error(`unexpected request ${url}`);
  });
}

describe("central de casos", () => {
  beforeEach(() => { vi.stubGlobal("EventSource", undefined); mockApi(); });
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });

  it("mantém aviso, provider e seletor de persona no cabeçalho", async () => {
    render(<App />);
    expect(await screen.findByText(config.warning)).not.toBeNull();
    expect(screen.getByText(/Groq principal/i)).not.toBeNull();
    expect((screen.getByLabelText("Persona simulada") as HTMLSelectElement).value).toBe("usr_1");
  });

  it("troca a área esquerda pelo menu fixo da direita sem perder o caso", async () => {
    render(<App />);
    await screen.findByText("TKT-1");
    fireEvent.click(screen.getByText("Contexto", { selector: "button" }));
    expect(screen.getByText("Contexto do caso", { selector: "h1" })).not.toBeNull();
    expect(screen.getByText("asset_1")).not.toBeNull();
    expect(screen.getByTestId("right-menu").classList.contains("right-rail")).toBe(true);
  });

  it("explica o bloqueio e atualiza capacidades ao trocar persona", async () => {
    render(<App />);
    await screen.findByText("TKT-1");
    fireEvent.click(screen.getByText("Decisões", { selector: "button" }));
    expect(screen.getByText(/sem decisões permitidas/i)).not.toBeNull();
    fireEvent.change(screen.getByLabelText("Persona simulada"), { target: { value: "boss_1" } });
    await waitFor(() => expect(screen.getAllByText(/autoridade da empresa/i).length).toBeGreaterThan(1));
  });
});
