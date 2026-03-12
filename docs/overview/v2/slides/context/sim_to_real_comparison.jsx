import { useState } from "react";

const sacSteps = [
  {
    id: 1,
    title: "Build Simulation Environment",
    detail: "Create a virtual physics simulator of the robot.",
    type: "shared",
  },
  {
    id: 2,
    title: "Massive Parallel RL Training",
    detail: "Run thousands of simulated robots simultaneously to learn a policy fast.",
    type: "shared",
  },
  {
    id: 3,
    title: "Domain Randomization",
    detail: "Randomly vary friction, mass, gravity etc. so the policy generalises to the real world.",
    type: "shared",
  },
  {
    id: 4,
    title: "Trained RL Policy",
    detail: "Policy has converged in simulation.",
    type: "shared",
  },
  {
    id: 5,
    title: "Export Policy (TorchScript / ONNX)",
    detail: "Freeze weights and export to a deployable format.",
    type: "shared",
  },
  {
    id: 6,
    title: "Deploy on Real Robot",
    detail: "Load the exported policy onto the physical robot.",
    type: "shared",
  },
  {
    id: 7,
    title: "Robot Control Loop\nObserve → Policy → Action",
    detail: "Standard RL loop running in real time on the robot.",
    type: "shared",
  },
  {
    id: 8,
    title: "Safety Wrapper\nTorque / Velocity / Joint Limits",
    detail: "Hard-coded safety limits before any action reaches motors.",
    type: "shared",
  },
  {
    id: 9,
    title: "Real World Execution",
    detail: "The robot physically interacts with the world.",
    type: "shared",
  },
  {
    id: 10,
    title: "Collect Real Robot Logs",
    detail: "Record all real-world interactions for fine-tuning.",
    type: "shared",
  },
  {
    id: 11,
    title: "Offline RL Fine-Tuning (SAC)",
    detail: "SAC retrains on the collected real-world data.\n\n⚠️ Problem: SAC trains on a growing replay buffer weighted toward recent real-world data. It gradually forgets the simulation knowledge. If real-world conditions change (new floor, heavier payload), it may catastrophically forget earlier adaptations too.\n\n⚠️ Problem: SAC is off-policy and sensitive to hyperparameters — unstable in deployment.",
    type: "sac",
    warning: true,
  },
  {
    id: 12,
    title: "Improved Policy",
    detail: "Re-exported and redeployed. But quality degrades over time as simulation knowledge is forgotten.",
    type: "sac",
    warning: true,
  },
];

const lcpoSteps = [
  {
    id: 1,
    title: "Build Simulation Environment",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 2,
    title: "Massive Parallel RL Training",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 3,
    title: "Domain Randomization",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 4,
    title: "Trained RL Policy",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 5,
    title: "Export Policy (TorchScript / ONNX)",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 6,
    title: "Deploy on Real Robot",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 7,
    title: "Robot Control Loop\nObserve → Policy → Action",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 8,
    title: "Safety Wrapper\nTorque / Velocity / Joint Limits",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 9,
    title: "Real World Execution",
    detail: "Same as SAC pipeline.",
    type: "shared",
  },
  {
    id: 10,
    title: "Collect Real Robot Logs\n+ Reservoir Sample into Bₐ",
    detail: "Record real-world interactions AND store a random reservoir sample into LCPO's old buffer Bₐ.\n\n✅ Bₐ contains both simulation experiences AND real-world experiences — proportionally sampled.",
    type: "lcpo",
    highlight: true,
  },
  {
    id: 11,
    title: "OOD Detection\nW(Bₐ, Bᵣ)",
    detail: "Before each policy update, find anchor samples in Bₐ whose context (e.g. floor friction, payload) is sufficiently different from the current real-world batch.\n\n✅ This automatically catches: sim vs real differences, changing real-world conditions, different robot configurations.",
    type: "lcpo",
    highlight: true,
  },
  {
    id: 12,
    title: "Online LCPO Fine-Tuning",
    detail: "LCPO updates the policy on current real-world experiences, while constraining it to not change outputs on OOD anchor samples.\n\n✅ Learns from new real-world data\n✅ Remembers simulation knowledge\n✅ Remembers earlier real-world adaptations\n✅ On-policy: stable, no bootstrapping issues\n✅ No task labels needed — works with smooth real-world drift",
    type: "lcpo",
    highlight: true,
  },
  {
    id: 13,
    title: "Continuously Improved Policy",
    detail: "Policy improves over time without forgetting. No need to re-export — LCPO runs live on the robot.\n\n✅ Sim knowledge preserved\n✅ Adapts to changing real-world conditions\n✅ Stable long-term performance",
    type: "lcpo",
    highlight: true,
  },
];

const diffs = [
  {
    aspect: "Fine-tuning approach",
    sac: "Offline — collect logs first, then retrain separately",
    lcpo: "Online — learns continuously while deployed",
  },
  {
    aspect: "Catastrophic forgetting",
    sac: "⚠️ Forgets simulation knowledge over time as real-world data dominates buffer",
    lcpo: "✅ Anchors on sim experiences in Bₐ, never forgets them",
  },
  {
    aspect: "Changing real-world conditions",
    sac: "⚠️ Fine-tuning on new conditions overwrites old adaptations",
    lcpo: "✅ OOD detector catches context shifts, anchors on prior adaptations",
  },
  {
    aspect: "Stability",
    sac: "⚠️ Off-policy bootstrapping causes instability, hyperparameter sensitive",
    lcpo: "✅ On-policy, more stable by design",
  },
  {
    aspect: "Task labels required?",
    sac: "Not explicitly, but assumes stationarity within each fine-tune round",
    lcpo: "✅ No task labels needed — works with arbitrary context drift",
  },
  {
    aspect: "Deployment model",
    sac: "Stop → collect → retrain → redeploy cycle",
    lcpo: "Continuous — never needs to stop or redeploy",
  },
];

function Pipeline({ steps, color, label }) {
  const [active, setActive] = useState(null);

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{
        textAlign: "center",
        marginBottom: 20,
        fontSize: 13,
        letterSpacing: 3,
        textTransform: "uppercase",
        color,
        fontFamily: "monospace",
        borderBottom: `2px solid ${color}`,
        paddingBottom: 10,
      }}>{label}</div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
        {steps.map((step, idx) => {
          const isActive = active === step.id;
          const isShared = step.type === "shared";
          const stepColor = isShared ? "#475569" : color;
          const bg = isShared ? "#0f1923" : (step.warning ? "#1a0a0a" : "#0a1a12");
          const borderColor = isActive ? stepColor : (isShared ? "#1e293b" : `${stepColor}44`);

          return (
            <div key={step.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
              {idx > 0 && (
                <div style={{ width: 2, height: 12, background: isShared ? "#1e293b" : `${stepColor}44` }} />
              )}
              <div
                onClick={() => setActive(isActive ? null : step.id)}
                style={{
                  width: "100%",
                  background: isActive ? (isShared ? "#0f1923" : bg) : (isShared ? "#0a0f18" : bg),
                  border: `1.5px solid ${borderColor}`,
                  borderRadius: 8,
                  padding: "10px 14px",
                  cursor: "pointer",
                  transition: "all 0.15s",
                  boxShadow: isActive && !isShared ? `0 0 12px ${stepColor}33` : "none",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: stepColor,
                    flexShrink: 0,
                  }} />
                  <span style={{
                    fontSize: 12,
                    color: isShared ? "#64748b" : (step.warning ? "#f87171" : step.highlight ? color : "#94a3b8"),
                    whiteSpace: "pre-wrap",
                    lineHeight: 1.4,
                    fontWeight: !isShared ? "bold" : "normal",
                  }}>{step.title}</span>
                  {!isShared && (
                    <span style={{ marginLeft: "auto", fontSize: 10, color: stepColor }}>
                      {isActive ? "▲" : "▼"}
                    </span>
                  )}
                </div>

                {isActive && step.detail && (
                  <div style={{
                    marginTop: 10,
                    paddingTop: 10,
                    borderTop: `1px solid ${stepColor}33`,
                    fontSize: 11,
                    color: "#94a3b8",
                    lineHeight: 1.7,
                    whiteSpace: "pre-wrap",
                    borderLeft: `3px solid ${stepColor}`,
                    paddingLeft: 10,
                  }}>
                    {step.detail}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function Comparison() {
  const [tab, setTab] = useState("diagram");

  return (
    <div style={{
      background: "#060910",
      minHeight: "100vh",
      fontFamily: "'Georgia', serif",
      color: "#e2e8f0",
      padding: "32px 16px",
    }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ fontSize: 10, letterSpacing: 6, color: "#475569", textTransform: "uppercase", fontFamily: "monospace", marginBottom: 10 }}>
            Sim-to-Real Pipeline
          </div>
          <h1 style={{ fontSize: 26, fontWeight: "normal", margin: 0, color: "#f8fafc", letterSpacing: -0.5 }}>
            SAC vs LCPO Fine-Tuning
          </h1>
          <p style={{ color: "#475569", fontSize: 13, fontStyle: "italic", marginTop: 6 }}>
            Click highlighted steps to see what changes
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 28 }}>
          {[["diagram", "Side-by-Side Diagram"], ["diff", "Key Differences"]].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              style={{
                background: tab === key ? "#1e293b" : "transparent",
                border: `1px solid ${tab === key ? "#475569" : "#1e293b"}`,
                borderRadius: 6,
                padding: "8px 18px",
                color: tab === key ? "#f1f5f9" : "#64748b",
                cursor: "pointer",
                fontSize: 13,
                fontFamily: "Georgia, serif",
              }}
            >{label}</button>
          ))}
        </div>

        {tab === "diagram" && (
          <div style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
            <Pipeline steps={sacSteps} color="#f87171" label="With SAC" />
            <div style={{ width: 1, background: "#1e293b", alignSelf: "stretch", flexShrink: 0 }} />
            <Pipeline steps={lcpoSteps} color="#34d399" label="With LCPO" />
          </div>
        )}

        {tab === "diff" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {diffs.map((d, i) => (
              <div key={i} style={{
                background: "#0a0f18",
                border: "1px solid #1e293b",
                borderRadius: 10,
                overflow: "hidden",
              }}>
                <div style={{
                  background: "#111827",
                  padding: "8px 16px",
                  fontSize: 12,
                  letterSpacing: 1,
                  color: "#94a3b8",
                  textTransform: "uppercase",
                  fontFamily: "monospace",
                  borderBottom: "1px solid #1e293b",
                }}>{d.aspect}</div>
                <div style={{ display: "flex" }}>
                  <div style={{
                    flex: 1,
                    padding: "12px 16px",
                    borderRight: "1px solid #1e293b",
                    fontSize: 13,
                    color: "#fca5a5",
                    lineHeight: 1.6,
                  }}>
                    <div style={{ fontSize: 10, color: "#f87171", letterSpacing: 2, marginBottom: 6, fontFamily: "monospace" }}>SAC</div>
                    {d.sac}
                  </div>
                  <div style={{
                    flex: 1,
                    padding: "12px 16px",
                    fontSize: 13,
                    color: "#6ee7b7",
                    lineHeight: 1.6,
                  }}>
                    <div style={{ fontSize: 10, color: "#34d399", letterSpacing: 2, marginBottom: 6, fontFamily: "monospace" }}>LCPO</div>
                    {d.lcpo}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Legend */}
        <div style={{
          marginTop: 32,
          padding: "14px 18px",
          background: "#0a0f18",
          borderRadius: 8,
          border: "1px solid #1e293b",
          display: "flex",
          gap: 24,
          flexWrap: "wrap",
          fontSize: 12,
          color: "#64748b",
        }}>
          <span><span style={{ color: "#475569" }}>●</span> Shared step (same in both)</span>
          <span><span style={{ color: "#f87171" }}>●</span> SAC-specific step (click to expand)</span>
          <span><span style={{ color: "#34d399" }}>●</span> LCPO-specific step (click to expand)</span>
        </div>
      </div>
    </div>
  );
}
