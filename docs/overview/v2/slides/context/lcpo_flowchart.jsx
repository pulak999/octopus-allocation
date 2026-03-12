import { useState } from "react";

const steps = [
  {
    id: 1,
    line: "Line 1",
    title: "Initialize",
    short: "Set up empty buffers & random weights",
    math: "θ₀ = random weights\nBₐ = {} (empty)",
    plain: "Create the actor+critic neural networks with random weights. Start with an empty 'memory buffer' for old experiences.",
    color: "#6366f1",
    icon: "⚙️",
  },
  {
    id: 2,
    line: "Line 3",
    title: "Collect Fresh Experiences",
    short: "Play in the environment for 200 steps",
    math: "Bᵣ ← {(sₜ, zₜ, aₜ, rₜ, sₜ₊₁)}",
    plain: "The agent plays for 200 steps right now, collecting:\n• sₜ = current state (e.g. robot position)\n• zₜ = current context (e.g. wind strength)\n• aₜ = action taken\n• rₜ = reward received\n• sₜ₊₁ = next state",
    color: "#0ea5e9",
    icon: "🎮",
  },
  {
    id: 3,
    line: "Line 4",
    title: "Find OOD Anchor Samples",
    short: "Find old experiences from different contexts",
    math: "Sᶜ ← W(Bₐ, Bᵣ)\n\nKeep samples where:\n‖zᵢ - avg(zᵣ)‖ > σ",
    plain: "Look through old memory buffer Bₐ. Keep only experiences whose context zᵢ is far from the current context average.\n\nThese are our 'anchors' — things we must not forget.",
    color: "#f59e0b",
    icon: "🔍",
  },
  {
    id: 4,
    line: "Line 5",
    title: "Compute Learning Gradient",
    short: "Which direction improves performance right now?",
    math: "v ← ∇θ Ltot(θ; Bᵣ)|θ₀\n\nLtot = policy loss + entropy loss",
    plain: "Compute the standard A2C gradient — the direction to move the neural network weights to get better rewards on the current batch.\n\nThis is the 'naive' update direction before we apply any constraints.",
    color: "#10b981",
    icon: "📐",
  },
  {
    id: "branch",
    type: "branch",
    question: "Are there OOD anchor samples?",
    yes: "Yes → Apply constraint",
    no: "No → Normal update",
    color: "#8b5cf6",
    icon: "🔀",
  },
  {
    id: 5,
    line: "Lines 7–8",
    title: "Constrained Update Direction",
    short: "Adjust gradient to respect anchors",
    math: "g(x) = curvature of KL constraint\nvᶜ ← conjgrad(v, g(·))\n\nGoal: find vᶜ that:\n• stays close to v (still learns)\n• keeps KL(π_old ‖ π_new; Sᶜ) ≤ c_anchor",
    plain: "The conjugate gradient method finds the BEST update direction that:\n1. Still improves performance on current experiences (like v)\n2. Does NOT change policy outputs on the OOD anchor samples\n\nThink of it as: 'learn as much as possible without forgetting old contexts'",
    color: "#ec4899",
    icon: "🧮",
    branch: "yes",
  },
  {
    id: 6,
    line: "Lines 9–10",
    title: "Line Search: Shrink if Needed",
    short: "Halve step size until constraints are satisfied",
    math: "while θ_old + vᶜ violates constraints:\n    vᶜ ← vᶜ / 2\n\nTwo constraints must hold:\n① KL(π; Sᶜ) ≤ c_anchor  ← don't forget old\n② KL(π; Bᵣ) ≤ c_recent  ← don't destabilize now",
    plain: "Even after conjugate gradient, the step might be too big. So we keep halving it until both constraints are satisfied:\n\n① Anchor constraint: policy doesn't change much on OOD samples\n② TRPO constraint: policy doesn't change too wildly on current samples either (stability)",
    color: "#f97316",
    icon: "📏",
    branch: "yes",
  },
  {
    id: 7,
    line: "Line 11 / Line 13",
    title: "Apply the Update",
    short: "Move the neural network weights",
    math: "If OOD samples existed:\n  θ₀ ← θ₀ + vᶜ  (constrained)\n\nIf no OOD samples:\n  θ₀ ← θ₀ + v   (unconstrained)",
    plain: "Update the policy network weights:\n\n• If we found OOD anchors → use the constrained direction vᶜ\n• If no OOD anchors found (e.g. early training, or context hasn't changed much) → just do normal gradient descent with v",
    color: "#14b8a6",
    icon: "✅",
  },
  {
    id: 8,
    line: "Line 14",
    title: "Update Memory Buffer",
    short: "Add new experiences to Bₐ via reservoir sampling",
    math: "Bₐ ← Bₐ + Bᵣ\n\nReservoir sampling:\nP(any sample in Bₐ) = nᵦ / nₛ",
    plain: "Add the current experiences to the memory buffer using reservoir sampling. This ensures every past interaction has an EQUAL chance of being remembered — old contexts don't get pushed out systematically.\n\nThen go back to step 2 and repeat forever.",
    color: "#6366f1",
    icon: "💾",
  },
];

export default function LCPOFlowchart() {
  const [active, setActive] = useState(null);

  const mainSteps = steps.filter((s) => s.type !== "branch");
  const branch = steps.find((s) => s.type === "branch");

  return (
    <div style={{
      background: "#0a0a0f",
      minHeight: "100vh",
      fontFamily: "'Georgia', serif",
      color: "#e2e8f0",
      padding: "40px 20px",
    }}>
      <div style={{ maxWidth: 700, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 48 }}>
          <div style={{
            fontSize: 11,
            letterSpacing: 6,
            color: "#6366f1",
            textTransform: "uppercase",
            marginBottom: 12,
            fontFamily: "monospace",
          }}>Algorithm 1</div>
          <h1 style={{
            fontSize: 32,
            fontWeight: "normal",
            margin: 0,
            color: "#f8fafc",
            letterSpacing: -1,
          }}>LCPO Training</h1>
          <p style={{ color: "#64748b", marginTop: 8, fontSize: 14, fontStyle: "italic" }}>
            Click any step to expand the explanation
          </p>
        </div>

        {/* Steps */}
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {steps.map((step, idx) => {
            if (step.type === "branch") {
              return (
                <div key="branch" style={{ display: "flex", flexDirection: "column", alignItems: "center", margin: "4px 0" }}>
                  {/* connector in */}
                  <div style={{ width: 2, height: 20, background: "#334155" }} />
                  <div style={{
                    background: "#1e1b4b",
                    border: "2px solid #4f46e5",
                    borderRadius: 8,
                    padding: "14px 28px",
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    fontSize: 14,
                    color: "#a5b4fc",
                  }}>
                    <span style={{ fontSize: 20 }}>{step.icon}</span>
                    <div>
                      <div style={{ fontWeight: "bold", color: "#c7d2fe" }}>{step.question}</div>
                      <div style={{ fontSize: 12, marginTop: 4, color: "#6366f1" }}>
                        <span style={{ marginRight: 16 }}>✓ {step.yes}</span>
                        <span>✗ {step.no}</span>
                      </div>
                    </div>
                  </div>
                  {/* connector out */}
                  <div style={{ width: 2, height: 20, background: "#334155" }} />
                </div>
              );
            }

            const isActive = active === step.id;
            const isBranchStep = step.branch === "yes";

            return (
              <div key={step.id} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                {idx > 0 && <div style={{ width: 2, height: isBranchStep ? 4 : 16, background: "#334155" }} />}

                {/* Step card */}
                <div
                  onClick={() => setActive(isActive ? null : step.id)}
                  style={{
                    width: "100%",
                    background: isActive ? "#0f172a" : "#111827",
                    border: `2px solid ${isActive ? step.color : "#1e293b"}`,
                    borderRadius: 12,
                    padding: "16px 20px",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    boxShadow: isActive ? `0 0 20px ${step.color}33` : "none",
                  }}
                >
                  {/* Header row */}
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{
                      width: 40,
                      height: 40,
                      borderRadius: 8,
                      background: `${step.color}22`,
                      border: `1px solid ${step.color}44`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 18,
                      flexShrink: 0,
                    }}>
                      {step.icon}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{
                          fontSize: 10,
                          fontFamily: "monospace",
                          color: step.color,
                          background: `${step.color}15`,
                          padding: "2px 8px",
                          borderRadius: 4,
                          letterSpacing: 1,
                        }}>{step.line}</span>
                        <span style={{ fontSize: 15, fontWeight: "bold", color: "#f1f5f9" }}>{step.title}</span>
                      </div>
                      <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 3 }}>{step.short}</div>
                    </div>
                    <div style={{
                      color: "#475569",
                      fontSize: 12,
                      transform: isActive ? "rotate(180deg)" : "rotate(0)",
                      transition: "transform 0.2s",
                    }}>▼</div>
                  </div>

                  {/* Expanded content */}
                  {isActive && (
                    <div style={{ marginTop: 16, borderTop: "1px solid #1e293b", paddingTop: 16 }}>
                      {/* Math block */}
                      <div style={{
                        background: "#0d1117",
                        border: "1px solid #21262d",
                        borderRadius: 8,
                        padding: "12px 16px",
                        fontFamily: "monospace",
                        fontSize: 13,
                        color: "#79c0ff",
                        whiteSpace: "pre-wrap",
                        marginBottom: 12,
                        lineHeight: 1.8,
                      }}>
                        {step.math}
                      </div>
                      {/* Plain explanation */}
                      <div style={{
                        fontSize: 13,
                        color: "#cbd5e1",
                        lineHeight: 1.8,
                        whiteSpace: "pre-wrap",
                        background: `${step.color}08`,
                        borderLeft: `3px solid ${step.color}`,
                        padding: "10px 14px",
                        borderRadius: "0 6px 6px 0",
                      }}>
                        {step.plain}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {/* Loop back arrow */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div style={{ width: 2, height: 16, background: "#334155" }} />
            <div style={{
              border: "2px dashed #334155",
              borderRadius: 8,
              padding: "10px 20px",
              color: "#475569",
              fontSize: 13,
              fontStyle: "italic",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}>
              <span>🔁</span> Repeat for every iteration until deployment ends
            </div>
          </div>
        </div>

        {/* Legend */}
        <div style={{
          marginTop: 48,
          padding: 20,
          background: "#0f172a",
          borderRadius: 12,
          border: "1px solid #1e293b",
        }}>
          <div style={{ fontSize: 11, letterSpacing: 4, color: "#475569", textTransform: "uppercase", marginBottom: 14, fontFamily: "monospace" }}>Key Symbols</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 13 }}>
            {[
              ["θ", "Neural network weights (what gets updated)"],
              ["Bᵣ", "Recent batch — fresh experiences right now"],
              ["Bₐ", "Old buffer — all past experiences (reservoir sampled)"],
              ["Sᶜ", "OOD anchor samples from Bₐ"],
              ["v", "Raw gradient direction (unconstrained)"],
              ["vᶜ", "Constrained gradient direction (LCPO's output)"],
              ["KL divergence", "How much the policy changed on a set of samples"],
              ["c_anchor", "Max allowed policy change on OOD samples"],
            ].map(([sym, desc]) => (
              <div key={sym} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <span style={{ fontFamily: "monospace", color: "#79c0ff", flexShrink: 0, minWidth: 80 }}>{sym}</span>
                <span style={{ color: "#64748b" }}>{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
