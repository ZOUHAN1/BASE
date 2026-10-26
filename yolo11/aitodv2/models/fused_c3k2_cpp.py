import hashlib
import types

import torch
from torch.utils.cpp_extension import load_inline


SOURCE = r'''
#include <torch/extension.h>

torch::Tensor conv_silu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t padding) {
  auto convolved = at::conv2d(
      input, weight, bias, {1, 1}, {padding, padding}, {1, 1}, 1);
  return at::silu(convolved);
}

torch::Tensor conv_auto(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    bool activate) {
  const auto groups = input.size(1) / weight.size(1);
  const auto padding = weight.size(2) / 2;
  auto output = at::conv2d(
      input, weight, bias, {1, 1}, {padding, padding}, {1, 1}, groups);
  return activate ? at::silu(output) : output;
}

torch::Tensor c3k2_from_projected(
    const torch::Tensor& first,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  TORCH_CHECK(weights.size() == 9 && biases.size() == 9,
              "C3k2 fused eager op expects nine convolutions");
  auto halves = first.chunk(2, 1);

  auto main = conv_silu(halves[1], weights[2], biases[2], 0);
  auto residual = main;
  main = residual + conv_silu(
      conv_silu(main, weights[5], biases[5], 1), weights[6], biases[6], 1);
  residual = main;
  main = residual + conv_silu(
      conv_silu(main, weights[7], biases[7], 1), weights[8], biases[8], 1);

  auto bypass = conv_silu(halves[1], weights[3], biases[3], 0);
  auto inner = conv_silu(at::cat({main, bypass}, 1), weights[4], biases[4], 0);
  return conv_silu(
      at::cat({halves[0], halves[1], inner}, 1), weights[1], biases[1], 0);
}

torch::Tensor c3k2_forward(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  auto first = conv_silu(input, weights[0], biases[0], 0);
  return c3k2_from_projected(first, weights, biases);
}

torch::Tensor brr_refine_forward(
    const torch::Tensor& merged,
    const torch::Tensor& anchor,
    const torch::Tensor& semantic_weight,
    const torch::Tensor& anchor_weight,
    const torch::Tensor& first_bias,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  const std::vector<int64_t> stride{1, 1};
  const std::vector<int64_t> padding{0, 0};
  const std::vector<int64_t> dilation{1, 1};
  auto semantic = at::conv2d(
      merged, semantic_weight, c10::nullopt, stride, padding, dilation, 1);
  auto projected = at::conv2d(
      anchor, anchor_weight, first_bias, stride, padding, dilation, 1);
  auto first = at::silu(projected + semantic);
  return c3k2_from_projected(first, weights, biases);
}

torch::Tensor arr_refine_forward(
    const torch::Tensor& base,
    const torch::Tensor& aggregated,
    const torch::Tensor& gate,
    const torch::Tensor& correction_weight,
    const torch::Tensor& base_weight,
    const torch::Tensor& first_bias,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  const std::vector<int64_t> stride{1, 1};
  const std::vector<int64_t> padding{0, 0};
  const std::vector<int64_t> dilation{1, 1};
  auto correction = at::conv2d(
      aggregated, correction_weight, c10::nullopt, stride, padding, dilation, 1);
  auto projected = at::conv2d(
      base, base_weight, first_bias, stride, padding, dilation, 1);
  auto first = at::silu(projected + gate * correction);
  return c3k2_from_projected(first, weights, biases);
}

torch::Tensor detect_box_forward(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  TORCH_CHECK(weights.size() == 3 && biases.size() == 3,
              "Detect box branch expects three convolutions");
  auto output = conv_auto(input, weights[0], biases[0], true);
  output = conv_auto(output, weights[1], biases[1], true);
  return conv_auto(output, weights[2], biases[2], false);
}

torch::Tensor detect_class_forward(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  TORCH_CHECK(weights.size() == 5 && biases.size() == 5,
              "Detect class branch expects five convolutions");
  auto output = input;
  for (int64_t index = 0; index < 4; ++index) {
    output = conv_auto(output, weights[index], biases[index], true);
  }
  return conv_auto(output, weights[4], biases[4], false);
}

torch::Tensor sppf_forward(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases) {
  TORCH_CHECK(weights.size() == 2 && biases.size() == 2,
              "SPPF eager op expects two convolutions");
  auto first = conv_silu(input, weights[0], biases[0], 0);
  auto pooled1 = at::max_pool2d(first, {5, 5}, {1, 1}, {2, 2}, {1, 1}, false);
  auto pooled2 = at::max_pool2d(pooled1, {5, 5}, {1, 1}, {2, 2}, {1, 1}, false);
  auto pooled3 = at::max_pool2d(pooled2, {5, 5}, {1, 1}, {2, 2}, {1, 1}, false);
  return conv_silu(
      at::cat({first, pooled1, pooled2, pooled3}, 1),
      weights[1], biases[1], 0);
}

torch::Tensor c2psa_forward(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& weights,
    const std::vector<torch::Tensor>& biases,
    int64_t channels,
    int64_t num_heads,
    int64_t key_dim,
    double scale) {
  TORCH_CHECK(weights.size() == 7 && biases.size() == 7,
              "C2PSA eager op expects seven convolutions");
  auto projected = conv_silu(input, weights[0], biases[0], 0);
  auto halves = projected.split_with_sizes({channels, channels}, 1);
  auto residual = halves[1];
  const auto batch = residual.size(0);
  const auto height = residual.size(2);
  const auto width = residual.size(3);
  const auto positions = height * width;
  const auto head_dim = channels / num_heads;

  auto qkv = conv_auto(residual, weights[2], biases[2], false)
                 .view({batch, num_heads, key_dim * 2 + head_dim, positions});
  auto parts = qkv.split_with_sizes({key_dim, key_dim, head_dim}, 2);
  auto attention = at::matmul(parts[0].transpose(-2, -1), parts[1]) * scale;
  attention = at::softmax(attention, -1, c10::nullopt);
  auto value_image = parts[2].reshape({batch, channels, height, width});
  auto attended = at::matmul(parts[2], attention.transpose(-2, -1))
                      .view({batch, channels, height, width});
  attended = attended + conv_auto(value_image, weights[4], biases[4], false);
  residual = residual + conv_auto(attended, weights[3], biases[3], false);
  auto feed_forward = conv_silu(residual, weights[5], biases[5], 0);
  residual = residual + conv_auto(feed_forward, weights[6], biases[6], false);
  return conv_silu(at::cat({halves[0], residual}, 1), weights[1], biases[1], 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("c3k2_forward", &c3k2_forward,
             "C3k2 forward with eager ATen operators");
  module.def("brr_refine_forward", &brr_refine_forward,
             "Algebraically fused BRR output and C3k2 input projections");
  module.def("arr_refine_forward", &arr_refine_forward,
             "Algebraically fused ARR relay and C3k2 input projection");
  module.def("detect_box_forward", &detect_box_forward,
             "Detect box branch with eager ATen operators");
  module.def("detect_class_forward", &detect_class_forward,
             "Detect class branch with eager ATen operators");
  module.def("sppf_forward", &sppf_forward,
             "SPPF forward with eager ATen operators");
  module.def("c2psa_forward", &c2psa_forward,
             "C2PSA forward with eager ATen operators");
}
'''


_EXTENSION = None
_LOAD_ERROR = None


def extension():
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is None and _LOAD_ERROR is None:
        name = 'brsa_c3k2_' + hashlib.sha256(SOURCE.encode()).hexdigest()[:12]
        try:
            _EXTENSION = load_inline(
                name=name,
                cpp_sources=SOURCE,
                functions=None,
                extra_cflags=['-O3'],
                with_cuda=False,
                verbose=False,
            )
        except Exception as error:
            _LOAD_ERROR = error
    return _EXTENSION


def load_error():
    return _LOAD_ERROR


def _conv_parameters(module):
    convolution = module.conv
    if convolution.bias is None:
        raise RuntimeError('C3k2 C++ fusion requires Conv-BN fusion first')
    return convolution.weight, convolution.bias


def bind_fast_c3k2(module):
    inner = module.m[0]
    bottlenecks = list(inner.m)
    if len(bottlenecks) != 2:
        raise RuntimeError('C3k2 C++ fusion expects two inner bottlenecks')
    convolutions = [
        module.cv1,
        module.cv2,
        inner.cv1,
        inner.cv2,
        inner.cv3,
        bottlenecks[0].cv1,
        bottlenecks[0].cv2,
        bottlenecks[1].cv1,
        bottlenecks[1].cv2,
    ]
    parameters = [_conv_parameters(child) for child in convolutions]
    module._fast_weights = tuple(item[0] for item in parameters)
    module._fast_biases = tuple(item[1] for item in parameters)
    module._native_forward = module.forward

    implementation = extension()

    def forward(self, x):
        return implementation.c3k2_forward(x, self._fast_weights, self._fast_biases)

    module.forward = types.MethodType(forward, module)
    return module


def _plain_parameters(module):
    if module.bias is None:
        raise RuntimeError('Detect C++ fusion requires biased convolutions')
    return module.weight, module.bias


def _bind_branch(branch, convolutions, entrypoint):
    parameters = [(_conv_parameters(module) if hasattr(module, 'conv') else
                   _plain_parameters(module)) for module in convolutions]
    branch._fast_weights = tuple(item[0] for item in parameters)
    branch._fast_biases = tuple(item[1] for item in parameters)
    branch._native_forward = branch.forward

    implementation = extension()

    def forward(self, x):
        return getattr(implementation, entrypoint)(x, self._fast_weights, self._fast_biases)

    branch.forward = types.MethodType(forward, branch)


def bind_fast_detect(detect):
    for box in detect.cv2:
        _bind_branch(box, (box[0], box[1], box[2]), 'detect_box_forward')
    for classes in detect.cv3:
        convolutions = (
            classes[0][0], classes[0][1], classes[1][0], classes[1][1], classes[2])
        _bind_branch(classes, convolutions, 'detect_class_forward')
    return detect


def bind_fast_sppf(module):
    parameters = [_conv_parameters(child) for child in (module.cv1, module.cv2)]
    module._fast_weights = tuple(item[0] for item in parameters)
    module._fast_biases = tuple(item[1] for item in parameters)
    module._native_forward = module.forward

    implementation = extension()

    def forward(self, x):
        return implementation.sppf_forward(x, self._fast_weights, self._fast_biases)

    module.forward = types.MethodType(forward, module)
    return module


def bind_fast_c2psa(module):
    if len(module.m) != 1:
        raise RuntimeError('C2PSA C++ fusion expects one PSA block')
    block = module.m[0]
    attention = block.attn
    convolutions = (
        module.cv1,
        module.cv2,
        attention.qkv,
        attention.proj,
        attention.pe,
        block.ffn[0],
        block.ffn[1],
    )
    parameters = [_conv_parameters(child) for child in convolutions]
    module._fast_weights = tuple(item[0] for item in parameters)
    module._fast_biases = tuple(item[1] for item in parameters)
    module._fast_num_heads = attention.num_heads
    module._fast_key_dim = attention.key_dim
    module._fast_scale = attention.scale
    module._native_forward = module.forward

    implementation = extension()

    def forward(self, x):
        return implementation.c2psa_forward(
            x, self._fast_weights, self._fast_biases, self.c,
            self._fast_num_heads, self._fast_key_dim, self._fast_scale)

    module.forward = types.MethodType(forward, module)
    return module


def bind_fast_brr_refine(module):
    refine = module.refine
    if not hasattr(refine, '_fast_weights'):
        raise RuntimeError('BRR fusion requires a bound C3k2 refinement block')
    module._fast_refine_cache = None
    implementation = extension()

    def forward(self, merged, anchor):
        first_weight = refine._fast_weights[0]
        output_weight = self.output.weight
        cache_key = (first_weight.device, first_weight.dtype,
                     first_weight.data_ptr(), output_weight.data_ptr())
        cache = self._fast_refine_cache
        if cache is None or cache[0] != cache_key:
            semantic_mix = first_weight[:, :512, 0, 0]
            output_mix = output_weight[:, :, 0, 0]
            semantic_weight = torch.mm(semantic_mix, output_mix).reshape(
                first_weight.shape[0], output_weight.shape[1], 1, 1)
            anchor_weight = first_weight[:, 512:].contiguous()
            cache = (cache_key, semantic_weight, anchor_weight)
            self._fast_refine_cache = cache
        return implementation.brr_refine_forward(
            merged, anchor, cache[1], cache[2], refine._fast_biases[0],
            refine._fast_weights, refine._fast_biases)

    module._fast_refine_forward = types.MethodType(forward, module)
    return module


def bind_fast_arr_refine(module):
    refine = module.refine
    if not hasattr(refine, '_fast_weights'):
        raise RuntimeError('ARR fusion requires a bound C3k2 refinement block')
    module._fast_refine_cache = None
    implementation = extension()

    def forward(self, base, aggregated, gate):
        first_weight = refine._fast_weights[0]
        value_weight = self.value.weight
        cache_key = (first_weight.device, first_weight.dtype,
                     first_weight.data_ptr(), value_weight.data_ptr())
        cache = self._fast_refine_cache
        if cache is None or cache[0] != cache_key:
            first_mix = first_weight[:, :, 0, 0]
            value_mix = value_weight[:, :, 0, 0]
            correction_weight = torch.mm(first_mix, value_mix).reshape(
                first_weight.shape[0], value_weight.shape[1], 1, 1)
            base_weight = first_weight.contiguous()
            cache = (cache_key, correction_weight, base_weight)
            self._fast_refine_cache = cache
        return implementation.arr_refine_forward(
            base, aggregated, gate, cache[1], cache[2], refine._fast_biases[0],
            refine._fast_weights, refine._fast_biases)

    module._fast_refine_forward = types.MethodType(forward, module)
    return module


