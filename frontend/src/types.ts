export type Panel = "chat" | "cases" | "context" | "evidence" | "timeline" | "decisions" | "simulation";
export interface DemoConfig { mode: "live"; warning: string; industrial_api: string; primary_provider: string; fallback_provider: string; slack_configured: boolean }
export interface Persona { id: string; name: string; profile: "requester" | "tractian" | "authority"; company_id: string | null; permissions: string[] }
export interface DemoCase { id: string; ticket_id: string; company_id: string; requester_id: string; asset_id: string; initial_message: string; source_case_id: string | null; immutable: boolean; created_at: string }
export interface Message { id: string; case_id: string; persona_id: string; role: "user" | "assistant" | "system"; content: string; created_at: string }
export interface Execution { id: string; status: "queued" | "running" | "waiting_human" | "completed" | "failed"; provider: string | null; fallback_reason: string | null; trace_id: string | null; error_code: string | null }
export interface CaseDetail { case: DemoCase; messages: Message[]; executions: Execution[] }
export interface CaseEvent { id: number; kind: string; payload: Record<string, unknown>; created_at: string }
export interface Decision { id: string; case_id: string; audience: string; kind: string; status: string; summary: string; allowed_operations: string[]; expires_at: string }
