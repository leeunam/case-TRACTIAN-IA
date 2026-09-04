import { expect, test } from "@playwright/test";

const demoCase = {
  id: "case_public_1", ticket_id: "TKT-1", company_id: "comp_1",
  requester_id: "usr_1", asset_id: "asset_1", initial_message: "Ajude a analisar",
  source_case_id: null, immutable: true, created_at: "2026-01-01T00:00:00Z",
};

test("menu lateral muda a área principal e preserva o caso", async ({ page }) => {
  await page.route("http://127.0.0.1:8100/v1/**", async (route) => {
    const url = route.request().url();
    let json: unknown = [];
    if (url.endsWith("/demo/config")) json = { mode: "live", warning: "Demonstração com dados e identidades simulados.", industrial_api: "configured", primary_provider: "groq", fallback_provider: "nvidia-nim", slack_configured: false };
    else if (url.endsWith("/personas")) json = [{ id: "usr_1", name: "Joana", profile: "requester", company_id: "comp_1", permissions: ["read"] }];
    else if (url.endsWith("/cases")) json = [demoCase];
    else if (url.endsWith("/cases/case_public_1")) json = { case: demoCase, messages: [], executions: [] };
    await route.fulfill({ json });
  });
  await page.goto("/");
  await expect(page.getByText("TKT-1")).toBeVisible();
  await page.getByRole("button", { name: "Contexto" }).click();
  await expect(page.getByRole("heading", { name: "Contexto do caso" })).toBeVisible();
  await expect(page.getByText("asset_1")).toBeVisible();
});

test("link do Slack abre diretamente a central de decisões", async ({ page }) => {
  await page.route("http://127.0.0.1:8100/v1/**", async (route) => {
    const url = route.request().url();
    let json: unknown = [];
    if (url.endsWith("/demo/config")) json = { mode: "live", warning: "Demonstração com dados e identidades simulados.", industrial_api: "configured", primary_provider: "groq", fallback_provider: "nvidia-nim", slack_configured: false };
    else if (url.endsWith("/personas")) json = [{ id: "usr_1", name: "Joana", profile: "requester", company_id: "comp_1", permissions: ["read"] }];
    else if (url.endsWith("/cases")) json = [demoCase];
    else if (url.endsWith("/cases/case_public_1")) json = { case: demoCase, messages: [], executions: [] };
    await route.fulfill({ json });
  });
  await page.goto("/?decision=decision_1");
  await expect(page.getByRole("heading", { name: "Decisões pendentes" })).toBeVisible();
});
