"""Canonical representative OpenVLA blocks used by the airplane benchmark."""

SELECTED_LLAMA_BLOCKS = (8, 16, 24, 32)


def validate_selected_blocks(layer_count: int) -> tuple[int, ...]:
    if layer_count < SELECTED_LLAMA_BLOCKS[-1]:
        raise ValueError(
            f"OpenVLA exposes {layer_count} Llama blocks; "
            f"block {SELECTED_LLAMA_BLOCKS[-1]} is required"
        )
    return SELECTED_LLAMA_BLOCKS
