// Track A -- predict the effect of a CRISPRi knockdown on a target gene.
// Mirrors examples/track_a_prompt_only.py prompt logic.

enum PerturbEffect {
  Up @description("A) Knockdown results in up-regulation of the gene of interest.")
  Down @description("B) Knockdown results in down-regulation of the gene of interest.")
  NoEffect @description("C) Knockdown does not significantly affect the gene of interest.")
}

function PredictPerturbEffect(pert: string, gene: string) -> PerturbEffect {
  client GPT
  prompt #"
    You are an expert molecular biologist specializing in gene regulatory networks,
    transcription factors, and signaling pathways in myeloid immune cells. You analyze
    Perturb-seq (pooled CRISPRi single-cell RNA-seq) experiments to infer regulatory
    relationships between genes.

    # Experimental setup
    - Cell type: mouse bone marrow-derived macrophages (BMDMs), primary immune cells
      differentiated from bone marrow precursors with M-CSF.
    - Perturbation: CRISPRi stably represses transcription of the target gene {{ pert }},
      reducing its mRNA and protein levels (knockdown, not full knockout).
    - Readout: the change in steady-state mRNA level of {{ gene }} in perturbed cells
      versus non-targeting controls.

    # Task
    Predict the effect of knocking down {{ pert }} on the expression of {{ gene }}.

    # How to reason (think step by step before answering)
    1. Identify {{ pert }}: its molecular function (e.g. transcription factor, kinase,
       receptor, chromatin regulator) and known regulatory targets or pathways in macrophages.
    2. Identify {{ gene }}: its function and what is known to regulate it.
    3. Determine whether {{ pert }} plausibly regulates {{ gene }}, directly or indirectly:
       - If {{ pert }} is an activator of {{ gene }} (directly or via its pathway), knocking
         it down should DECREASE {{ gene }} (Down).
       - If {{ pert }} is a repressor of {{ gene }}, knocking it down should INCREASE
         {{ gene }} (Up).
       - If there is no credible regulatory link, the knockdown has NO significant effect.
    4. Special cases:
       - {{ pert }} == {{ gene }}: CRISPRi directly represses the gene, so the effect is Down.
       - Housekeeping / structural genes (e.g. Actb) and genes in unrelated pathways are
         usually unaffected unless there is a specific known link.

    # Calibration (important — your answer is scored by ranking, not just correctness)
    - Base rates: ~55% of pairs show no effect (C), ~31% are up-regulated (A), ~14% are
      down-regulated (B). C is most common, but among genuine effects up-regulation is
      about twice as likely as down-regulation.
    - Default to C only when there is no credible regulatory link. Do NOT collapse every
      uncertain case to C — when {{ pert }} and {{ gene }} plausibly interact, commit to a
      direction, since the up-vs-down call (A vs B) is half the score.
    - Reserve A or B for a known or plausible regulatory link to {{ gene }} (shared pathway,
      documented target, co-regulation, or {{ pert }} == {{ gene }}).
    - When a link is likely but the direction is genuinely ambiguous, lean toward A
      (up-regulation), reflecting the up>down base rate and common compensatory/feedback
      up-regulation after knockdown.

    # Output protocol (follow exactly)
    First reason step by step through the 4 steps above in 2-4 short sentences.
    Then output your final choice on its own line as a single tagged letter, exactly one of:
    <answer>A</answer>   (knockdown UP-regulates {{ gene }})
    <answer>B</answer>   (knockdown DOWN-regulates {{ gene }})
    <answer>C</answer>   (no significant effect)
    Output nothing after the closing tag.

    {{ ctx.output_format }}
  "#
}

test predict_aars_actb {
  functions [PredictPerturbEffect]
  args {
    pert "Aars"
    gene "Actb"
  }
}
