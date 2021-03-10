from enum import Enum


class Layout(Enum):
    Masked = 0  # Example: Transformer or CrossEntropyLoss via padding_idx
    Packed = 1  # Example: EmbeddingBag
    PackedSequence = 2  # Restricted to RNN
    List = 3  # Fallback and default for quick creation
    Native = 4  # Specialized, compact layout used within C++ extension
