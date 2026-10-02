from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIN_AXIS = ROOT / "external" / "Pain-axis"
PAIN_AXIS_COMMIT = "4d75cd90e206ea962f7a9101e65c85efea56723b"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"


def released_pain_vector(pain_axis_name="Qwen_2.5_7B_instruct", layer=24):
    """The released Pain Axis S2 vector, as float32 numpy."""
    import torch
    d = torch.load(PAIN_AXIS / "results/3.2_pain_vectors/pain_vectors" / pain_axis_name / "pain_vectors.pt",
                   map_location="cpu", weights_only=False)
    assert int(d["layer"]) == layer, "the monitor layer should be the extraction layer"
    return d["s2_pain_vector"].float().numpy()
