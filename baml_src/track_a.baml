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
    You are an expert molecular biologist who studies how genes are related using Perturb-seq.

    Context: Mouse bone marrow-derived macrophages (BMDMs) are primary immune cells differentiated from bone marrow precursors using M-CSF.

    Question: If you knockdown {{ pert }} using CRISPRi in mouse BMDMs, what is the effect on {{ gene }}?

    Your answer must be one of:
    A) Knockdown of {{ pert }} results in up-regulation of {{ gene }}.
    B) Knockdown of {{ pert }} results in down-regulation of {{ gene }}.
    C) Knockdown of {{ pert }} does not significantly affect {{ gene }}.

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
