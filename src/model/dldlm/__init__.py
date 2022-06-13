from typing import TYPE_CHECKING

from transformers.file_utils import _LazyModule, is_torch_available

_import_structure = {
    "configuration_dldlm": ["DLDLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "DLDLMConfig"],
    "tokenization_dldlm": ["DLDLMTokenizer"]
}

if is_torch_available():
    _import_structure["modeling_dldlm"] = [
        "DLDLM_PRETRAINED_CONFIG_ARCHIVE_LIST",
        "DLDLMFullModel",
        "DLDLMForSequenceClassification",
        "DLDLMLMHeadModel",
        "DLDLMModel",
        "DLDLMPreTrainedModel",
        "load_tf_weights_in_dldlm",
    ]

if TYPE_CHECKING:
    from configuration_dldlm import DLDLM_PRETRAINED_CONFIG_ARCHIVE_MAP, DLDLMConfig
    from tokenization_dldlm import DLDLMTokenizer

    if is_torch_available():
        from modeling_dldlm import (
            DLDLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            DLDLMFullModel,
            DLDLMForSequenceClassification,
            DLDLMLMHeadModel,
            DLDLMModel,
            DLDLMPreTrainedModel,
            load_tf_weights_in_dldlm,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
