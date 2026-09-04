import { FormEvent, useEffect, useState } from "react";
import { api } from "./api";
import type { CaseDetail, CaseEvent, Decision, DemoCase, DemoConfig, Panel, Persona } from "./types";
import "./styles.css";

const navigation: { id: Panel; label: string; glyph: string }[] = [
  { id: "chat", label: "Chat", glyph: "⌁" },
  { id: "cases", label: "Casos", glyph: "▤" },
  { id: "context", label: "Contexto", glyph: "◎" },
  { id: "evidence", label: "Evidências", glyph: "◇" },
  { id: "timeline", label: "Timeline", glyph: "↳" },
  { id: "decisions", label: "Decisões", glyph: "✓" },
  { id: "simulation", label: "Simulação", glyph: "⚙" },
];

const profileLabel = { requester: "Solicitante", tractian: "Equipe TRACTIAN", authority: "Autoridade da empresa" };

export default function App() {
  const [config, setConfig] = useState<DemoConfig>();
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [personaId, setPersonaId] = useState("");
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [caseId, setCaseId] = useState("");
  const [detail, setDetail] = useState<CaseDetail>();
  const [panel, setPanel] = useState<Panel>(() =>
    new URLSearchParams(window.location.search).has("decision") ? "decisions" : "chat"
  );
  const [events, setEvents] = useState<CaseEvent[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [error, setError] = useState("");

  const refreshCase = async (id: string) => setDetail(await api.case(id));
  const refreshDecisions = async () => {
    if (personaId) setDecisions(await api.decisions(personaId));
  };
  useEffect(() => {
    Promise.all([api.config(), api.personas(), api.cases()])
      .then(([nextConfig, nextPersonas, nextCases]) => {
        setConfig(nextConfig); setPersonas(nextPersonas); setCases(nextCases);
        setPersonaId(nextPersonas[0]?.id ?? "");
        const first = nextCases[0]?.id ?? ""; setCaseId(first);
        if (first) void refreshCase(first);
      }).catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (
      !caseId || import.meta.env.MODE === "test" ||
      (typeof navigator !== "undefined" && navigator.userAgent.includes("jsdom")) ||
      typeof EventSource === "undefined"
    ) return;
    const source = new EventSource(api.eventUrl(caseId));
    const append = (message: Event) => {
      const event = JSON.parse((message as MessageEvent).data) as CaseEvent;
      setEvents((old) => old.some((item) => item.id === event.id) ? old : [...old, event]);
      if (event.kind === "agent.completed" || event.kind.startsWith("decision.")) {
        void refreshCase(caseId);
      }
    };
    ["execution.queued", "execution.running", "planner.started", "tools.completed", "agent.completed", "agent.failed", "decision.requested", "decision.approved", "decision.rejected"].forEach((name) => source.addEventListener(name, append));
    return () => source.close();
  }, [caseId]);

  useEffect(() => {
    if (panel === "decisions" && personaId) {
      api.decisions(personaId).then(setDecisions).catch((reason: Error) => setError(reason.message));
    }
  }, [panel, personaId]);

  const persona = personas.find((item) => item.id === personaId);
  const execution = detail?.executions.at(-1);
  const status = execution?.status ?? (detail?.case.immutable ? "modelo público" : "pronto");

  async function selectCase(id: string) {
    setCaseId(id); setDetail(await api.case(id)); setEvents([]); setPanel("chat");
  }
  async function duplicate() {
    if (!detail) return;
    const created = await api.duplicate(detail.case.id);
    setCases((old) => [created, ...old]); await selectCase(created.id);
  }

  if (!config) return <main className="loading">{error || "Preparando central de casos…"}</main>;

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark">T</span><div><b>TRACTIAN</b><small>Case Intelligence</small></div></div>
      <div className="case-heading"><span className="eyebrow">Caso ativo</span><strong>{detail?.case.ticket_id ?? "Selecione"}</strong><span className={`status status-${status}`}>{status.replace("_", " ")}</span></div>
      <div className="provider"><span className="pulse" /> Groq principal <span>→ {config.fallback_provider} fallback</span></div>
      <label className="persona">Persona
        <select aria-label="Persona simulada" value={personaId} onChange={(event) => setPersonaId(event.target.value)}>
          {personas.map((item) => <option key={item.id} value={item.id}>{item.name} · {profileLabel[item.profile]}</option>)}
        </select>
      </label>
    </header>
    <div className="demo-warning">{config.warning}</div>
    <main className="workspace">
      <section className="content">
        {error && <div className="error" role="alert">{error}<button onClick={() => setError("")}>fechar</button></div>}
        {detail && <PanelContent panel={panel} detail={detail} persona={persona} personas={personas} cases={cases} events={events} decisions={decisions} onSelect={selectCase} onDuplicate={duplicate} onCreated={(created) => { setCases((old) => [created, ...old]); void selectCase(created.id); }} onRefresh={() => refreshCase(detail.case.id)} onDecisionsRefresh={refreshDecisions} onError={setError} />}
      </section>
      <nav className="right-rail" data-testid="right-menu" aria-label="Áreas da central">
        <div className="rail-caption">Navegação</div>
        {navigation.map((item) => <button key={item.id} className={panel === item.id ? "active" : ""} aria-label={item.label} onClick={() => setPanel(item.id)}><span>{item.glyph}</span>{item.label}</button>)}
        <div className="rail-foot"><span className="pulse" /> Serviços locais</div>
      </nav>
    </main>
  </div>;
}

function PanelContent(props: { panel: Panel; detail: CaseDetail; persona?: Persona; personas: Persona[]; cases: DemoCase[]; events: CaseEvent[]; decisions: Decision[]; onSelect(id: string): void; onDuplicate(): void; onCreated(value: DemoCase): void; onRefresh(): void; onDecisionsRefresh(): Promise<void>; onError(message: string): void }) {
  const { panel, detail, persona } = props;
  if (panel === "chat") return <Chat detail={detail} persona={persona} onDuplicate={props.onDuplicate} onRefresh={props.onRefresh} onError={props.onError} />;
  if (panel === "cases") return <Page title="Central de casos" subtitle="Casos públicos são modelos somente leitura; duplique ou personalize uma cópia."><NewCaseForm key={detail.case.id} base={detail.case} personas={props.personas} onCreated={props.onCreated} onError={props.onError} /><div className="case-grid">{props.cases.map((item) => <button className={`case-card ${item.id === detail.case.id ? "selected" : ""}`} key={item.id} onClick={() => props.onSelect(item.id)}><span>{item.immutable ? "Público" : "Minha simulação"}</span><b>{item.ticket_id}</b><p>{item.initial_message}</p><small>{item.asset_id}</small></button>)}</div></Page>;
  if (panel === "context") return <Page title="Contexto do caso" subtitle="Escopo confiável usado pelo agente."><dl className="facts"><div><dt>Empresa</dt><dd>{detail.case.company_id}</dd></div><div><dt>Solicitante original</dt><dd>{detail.case.requester_id}</dd></div><div><dt>Ativo central</dt><dd>{detail.case.asset_id}</dd></div><div><dt>Thread</dt><dd>{detail.case.id}</dd></div></dl></Page>;
  if (panel === "evidence") {
    const tools = [...props.events].reverse().find((item) => item.kind === "tools.completed")?.payload.tool_names as string[] | undefined;
    const result = [...props.events].reverse().find((item) => item.kind === "agent.completed")?.payload;
    return <Page title="Evidências" subtitle="Resumo sanitizado; sem notas de juiz ou raciocínio interno.">{tools || result ? <dl className="facts"><div><dt>Tools consultadas</dt><dd>{tools?.join(", ") || "nenhuma"}</dd></div><div><dt>Evidências citadas</dt><dd>{String(result?.evidence_count ?? 0)}</dd></div><div><dt>Limitações</dt><dd>{String(result?.limitation_count ?? 0)}</dd></div><div><dt>Trace público</dt><dd>{String(result?.trace_id ?? "aguardando")}</dd></div></dl> : <Empty text="As evidências utilizadas aparecerão após a execução ao vivo." />}</Page>;
  }
  if (panel === "timeline") return <Page title="Timeline operacional" subtitle="Eventos persistidos e recuperáveis após reconexão.">{props.events.length ? <ol className="timeline">{props.events.map((event) => <li key={event.id}><b>{event.kind}</b><time>{new Date(event.created_at).toLocaleTimeString()}</time></li>)}</ol> : <Empty text="Nenhum evento novo nesta conexão." />}</Page>;
  if (panel === "decisions") return <Decisions persona={persona} decisions={props.decisions} onResolved={props.onDecisionsRefresh} onError={props.onError} />;
  return <Page title="Simulação" subtitle="O modo é definido ao criar a cópia e segue persistido no caso."><dl className="facts"><div><dt>Modo atual</dt><dd>{detail.case.simulation_mode ?? "standard"}</dd></div><div><dt>Seed</dt><dd>{detail.case.seed ?? "gerenciada pelo simulador"}</dd></div></dl></Page>;
}

function NewCaseForm({ base, personas, onCreated, onError }: { base: DemoCase; personas: Persona[]; onCreated(value: DemoCase): void; onError(message: string): void }) {
  const [open, setOpen] = useState(false); const [busy, setBusy] = useState(false);
  const [company, setCompany] = useState(base.company_id); const [requester, setRequester] = useState(base.requester_id);
  const [asset, setAsset] = useState(base.asset_id); const [message, setMessage] = useState(base.initial_message);
  const [mode, setMode] = useState("standard"); const [seed, setSeed] = useState("");
  async function submit(event: FormEvent) { event.preventDefault(); setBusy(true); try { const created = await api.create({ source_case_id: base.immutable ? base.id : base.source_case_id ?? undefined, company_id: company, requester_id: requester, asset_id: asset, message, simulation_mode: mode, ...(mode === "custom_seed" ? { seed } : {}) }); onCreated(created); setOpen(false); } catch (reason) { onError((reason as Error).message); } finally { setBusy(false); } }
  if (!open) return <button className="primary new-case" onClick={() => setOpen(true)}>Criar ou editar cópia</button>;
  return <form className="case-form" onSubmit={submit}><label>Empresa<input value={company} onChange={(event) => setCompany(event.target.value)} required /></label><label>Pessoa solicitante<select value={requester} onChange={(event) => { const person = personas.find((item) => item.id === event.target.value); setRequester(event.target.value); if (person?.company_id) setCompany(person.company_id); }}>{personas.filter((item) => item.profile !== "tractian").map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label><label>Ativo<input value={asset} onChange={(event) => setAsset(event.target.value)} required /></label><label>Modo<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="standard">Padrão determinístico</option><option value="complete">Dados completos</option><option value="degraded">Dados degradados</option><option value="custom_seed">Seed explícita</option></select></label>{mode === "custom_seed" && <label>Seed<input value={seed} onChange={(event) => setSeed(event.target.value)} required /></label>}<label className="wide">Mensagem<textarea value={message} onChange={(event) => setMessage(event.target.value)} required /></label><div className="wide"><button className="primary" disabled={busy}>{busy ? "Criando…" : "Criar caso"}</button> <button type="button" onClick={() => setOpen(false)}>Cancelar</button></div></form>;
}

function Chat({ detail, persona, onDuplicate, onRefresh, onError }: { detail: CaseDetail; persona?: Persona; onDuplicate(): void; onRefresh(): void; onError(message: string): void }) {
  const [text, setText] = useState(""); const [sending, setSending] = useState(false);
  const canTalk = !detail.case.immutable && persona?.id === detail.case.requester_id;
  async function submit(event: FormEvent) { event.preventDefault(); if (!canTalk || !text.trim()) return; setSending(true); try { await api.message(detail.case.id, persona!.id, text.trim()); setText(""); onRefresh(); } catch (reason) { onError((reason as Error).message); } finally { setSending(false); } }
  return <Page title="Conversa com o agente" subtitle="Planner e writer trabalham ao vivo; ações continuam sob política determinística.">
    <div className="chat-log"><article className="bubble user"><span>Solicitação original</span>{detail.case.initial_message}</article>{detail.messages.map((message) => <article key={message.id} className={`bubble ${message.role}`}><span>{message.role === "assistant" ? "Agente TRACTIAN" : "Solicitante"}</span>{message.content}</article>)}</div>
    {detail.case.immutable ? <div className="locked"><b>🔒 Caso público protegido</b><p>Duplique este modelo para preservar o conjunto original e iniciar o fluxo real.</p><button className="primary" onClick={onDuplicate}>Duplicar e conversar</button></div> :
      <form className="composer" onSubmit={submit}><textarea aria-label="Mensagem" value={text} onChange={(event) => setText(event.target.value)} placeholder={canTalk ? "Descreva o que aconteceu com o ativo…" : "Troque para a persona solicitante para conversar"} disabled={!canTalk || sending} /><button className="primary" disabled={!canTalk || sending}>{sending ? "Enfileirando…" : "Enviar"}</button></form>}
  </Page>;
}

function Decisions({ persona, decisions, onResolved, onError }: { persona?: Persona; decisions: Decision[]; onResolved(): Promise<void>; onError(message: string): void }) {
  const label = persona ? profileLabel[persona.profile] : "Persona";
  return <Page title="Decisões pendentes" subtitle={`${label}: a permissão é validada novamente no backend.`}>{decisions.length ? decisions.map((item) => <article className="decision" key={item.id}><span>{item.kind}</span><h3>{item.summary}</h3><p>Expira em {new Date(item.expires_at).toLocaleString()}</p><div>{item.allowed_operations.map((operation) => <button key={operation} onClick={() => api.resolve(item.id, persona!.id, operation as "approve" | "reject").then(onResolved).catch((reason: Error) => onError(reason.message))}>{operation === "approve" ? "Aprovar" : "Rejeitar"}</button>)}</div></article>) : <div className="locked"><b>🔒 Sem decisões permitidas para esta persona</b><p>Solicitantes, equipe TRACTIAN e autoridade da empresa enxergam caixas diferentes.</p></div>}</Page>;
}

function Page({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) { return <div className="page"><div className="page-title"><span className="eyebrow">Case workspace</span><h1>{title}</h1><p>{subtitle}</p></div>{children}</div>; }
function Empty({ text }: { text: string }) { return <div className="empty"><span>◇</span><p>{text}</p></div>; }
