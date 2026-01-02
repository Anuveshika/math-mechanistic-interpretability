def parse_cot_steps(cot_text):
    """
    Very simple step parser.
    Each sentence is treated as one reasoning step.
    """
    steps = [s.strip() for s in cot_text.split(".") if len(s.strip()) > 0]
    return steps
