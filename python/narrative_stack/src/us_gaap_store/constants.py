# Note: Debugging hardcoded value instead of using `torch.finfo(torch.float32).eps`
# Upside of hardcoding this is to keep it identical between PyTorch and NumPy code.
EPSILON = 1e-8

STAGE2_CATEGORY_STACKS: list[str] = [
    "credit::instant",
    "credit::duration",
    "debit::instant",
    "debit::duration",
    "none::instant",
    "none::duration",
]

# Assert all category strings are unique
assert len(STAGE2_CATEGORY_STACKS) == len(set(STAGE2_CATEGORY_STACKS)), (
    "`STAGE2_CATEGORY_STACKS` must contain unique entries"
)

# Assert all combinations of balance type and period type are covered
expected_combinations = {
    f"{b}::{p}"
    for b in ["credit", "debit", "none"]
    for p in ["instant", "duration"]
}
assert set(STAGE2_CATEGORY_STACKS) == expected_combinations, (
    "`STAGE2_CATEGORY_STACKS` must contain all balance/period combinations"
)
