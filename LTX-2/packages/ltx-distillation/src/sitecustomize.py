import torch


if not hasattr(torch.nn.functional, "rms_norm"):
    def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
        input_dtype = input.dtype
        input_float = input.float()
        variance = input_float.pow(2).mean(dim=-1, keepdim=True)
        output = input_float * torch.rsqrt(variance + eps)
        output = output.to(input_dtype)
        if weight is not None:
            output = output * weight
        return output

    torch.nn.functional.rms_norm = rms_norm


if not hasattr(torch.nn, "RMSNorm"):
    class RMSNorm(torch.nn.Module):
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(
                    torch.ones(self.normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("weight", None)

        def forward(self, input):
            return torch.nn.functional.rms_norm(
                input,
                self.normalized_shape,
                self.weight,
                self.eps,
            )

    torch.nn.RMSNorm = RMSNorm
