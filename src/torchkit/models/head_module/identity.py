from __future__ import annotations

from torch import nn, Tensor

from torchkit.models._spec_utils import normalize_spec_kwargs


class IdentityHead(nn.Module):
    """Identity head module that simply returns the input as output.
    This can be used when you want the output of the head module to be the same as the
    output of the feature adapter (or the backbone if no adapter is used).
    
    An example is: a segmentation head that takes the "reconstruction" feature from the
    backbone and directly outputs it as the segmentation mask logits without any additional processing.
    """

    def __init__(self, output_name: str = "output"):
        super().__init__()
        self.output_name = output_name
        self._spec_kwargs = normalize_spec_kwargs({"output_name": output_name})

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {self.output_name: x}

    def to_spec(self):
        from torchkit.models.head_module.factory import HeadModuleSpec

        return HeadModuleSpec(
            cls=self.__class__,
            kwargs=normalize_spec_kwargs(self._spec_kwargs),
        )
