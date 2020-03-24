// TODO:
// 1. &/const/optional params as params 
// 2. pass actual params instead of & to map
// 3. *input.data() vs input.data
// 4. fix TODOs
// 5. add better tests to check all optional params + fix code
// 6.

#include <seg_layers.h>
using namespace torch::nn;
namespace F = torch::nn::functional;

namespace torch {
namespace nested_tensor {
    //
    // relu
    //
    THPNestedTensor relu(const THPNestedTensor& input, bool inplace=false) {
        Tensor res = torch::relu(*input.data().contiguous().get_buffer());
        return THPNestedTensor(NestedTensor(std::move(res), input.data().nested_size()));
    }

    THPNestedTensor relu_out(THPNestedTensor& input) {
        input = relu(input, true);
        return input;
    }

    void add_relu(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), relu);
            m.def((name + std::string("_")).c_str(), relu_out);
    }

    //
    // dropout
    //
    THPNestedTensor dropout(const THPNestedTensor& input, double p, bool training, bool inplace) {
        Tensor res = torch::dropout(*input.data().contiguous().get_buffer(), p, training);
        return THPNestedTensor(NestedTensor(std::move(res), input.data().nested_size()));
    }

    void add_dropout(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), 
                  dropout,
                  py::arg("input"),
                  py::arg("p") = 0.5,
                  py::arg("training") = true,
                  py::arg("inplace") = false);
        }

    //
    // conv2d
    //
    THPNestedTensor conv2d(const THPNestedTensor& input, 
                           const Tensor weight, 
                           c10::optional<Tensor> bias, 
                           c10::optional<std::vector<int64_t>> stride,
                           c10::optional<std::vector<int64_t>> padding,
                           c10::optional<std::vector<int64_t>> dilation,
                           c10::optional<int64_t> groups) {
        NestedTensor nt = input.data().contiguous();
        TensorNode structure = nt.get_structure();
        TensorNode res = map([&](at::Tensor t){
            auto options = F::Conv2dFuncOptions().stride(stride.value())
                                                 .padding(padding.value())
                                                 .dilation(dilation.value())
                                                 .groups(groups.value());
            if (bias.has_value()) {
                return F::conv2d(t.unsqueeze(0), weight, options.bias(bias.value())).squeeze(0);
            } else {
                return F::conv2d(t.unsqueeze(0), weight, options).squeeze(0);
            }
        }, structure);

        return THPNestedTensor(NestedTensor(std::move(res)));
    }

    void add_conv2d(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
        m.def(name.c_str(), 
              conv2d,
              py::arg("input"), 
              py::arg("weight"),
              py::arg("bias") = nullptr,
              py::arg("stride") = std::vector<int64_t>({1, 1}),
              py::arg("padding") = std::vector<int64_t>({0, 0}),
              py::arg("dilation") = std::vector<int64_t>({1, 1}),
              py::arg("groups") = 1);
    }

    //
    // maxPool2d
    //
    THPNestedTensor maxPool2d(const THPNestedTensor& input,
                              std::vector<int64_t> kernel_size,
                              c10::optional<std::vector<int64_t>> stride,
                              c10::optional<std::vector<int64_t>> padding,
                              c10::optional<std::vector<int64_t>> dilation,
                              c10::optional<bool> return_indices, // TODO: enable this
                              c10::optional<bool> ceil_mode) {
        TensorNode structure = input.data().contiguous().get_structure();
        F::MaxPool2dFuncOptions options = F::MaxPool2dFuncOptions(kernel_size).stride(stride.value())
                                                                              .padding(padding.value())
                                                                              .dilation(dilation.value())
                                                                              //.return_indices(return_indices.value()),
                                                                              .ceil_mode(ceil_mode.value());
            
        TensorNode res = map([&, options](at::Tensor t){
            return F::max_pool2d(t, options);
        }, structure);

        return THPNestedTensor(NestedTensor(std::move(res)));
    }

    void add_max_pool2d(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), 
                  maxPool2d,
                  py::arg("input"), 
                  py::arg("kernel_size") = std::vector<int64_t>({}),
                  py::arg("stride") = std::vector<int64_t>({}),
                  py::arg("padding") = std::vector<int64_t>({0, 0}),
                  py::arg("dilation") = std::vector<int64_t>({1, 1}),
                  py::arg("return_indices") = false,
                  py::arg("ceil_mode") = false);
    }

    //
    // batch_norm
    //
    THPNestedTensor batch_norm(const THPNestedTensor& input,
                               const Tensor running_mean,
                               const Tensor running_var,
                               c10::optional<Tensor> weight,
                               c10::optional<Tensor> bias,
                               bool training, 
                               double momentum,
                               double eps) {
        TensorNode structure = input.data().contiguous().get_structure();
        auto options = F::BatchNormFuncOptions().weight(weight.value())
                                                .bias(bias.value())
                                                .momentum(momentum)
                                                .eps(eps)
                                                .training(training);

        TensorNode res = map([&, options, training](at::Tensor t){
            return F::batch_norm(t.unsqueeze(0), running_mean, running_var, options).squeeze(0);
        }, structure);

        return THPNestedTensor(NestedTensor(std::move(res)));
    }

    void add_batch_norm(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), 
                  batch_norm,
                  py::arg("input"),
                  py::arg("running_mean"),
                  py::arg("running_var"), 
                  py::arg("weight") = nullptr,
                  py::arg("bias") = nullptr,
                  py::arg("training") = false,
                  py::arg("momentum") = 0.1,
                  py::arg("eps") = 1e-05);
    }

    //
    // cross_entropy
    //
    THPNestedTensor cross_entropy(const THPNestedTensor& input,
                                  const THPNestedTensor& target,
                                  c10::optional<Tensor> weight,
                                  bool size_average, // TODO: use?
                                  int64_t ignore_index,
                                  bool reduce, // TODO: use?
                                  c10::optional<std::string> reduction) { 
        TensorNode input_structure = input.data().contiguous().get_structure();
        TensorNode target_structure = target.data().contiguous().get_structure();
        F::CrossEntropyFuncOptions::reduction_t redct;
        if (reduction.value() == "mean" || reduction.value() == "none") {
            redct = torch::kMean;
        } else if (reduction.value() == "sum") {
            redct = torch::kSum;
        } else {
            std::stringstream error;
            error << "Unexpected mode for reduction: " << reduction.value() << std::endl;
            throw error;
        }

        auto options = F::CrossEntropyFuncOptions().ignore_index(ignore_index).reduction(redct);

        TensorNode res = map([&, options] (at::Tensor input_tensor, at::Tensor target_tensor){
            return F::cross_entropy(input_tensor.unsqueeze(0), target_tensor.unsqueeze(0), options).squeeze(0);
        }, input_structure, target_structure);

        return THPNestedTensor(NestedTensor(std::move(res)));
    }

    void add_cross_entropy(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), 
                  cross_entropy,
                  py::arg("input"),
                  py::arg("target"),
                  py::arg("weight") = nullptr,
                  py::arg("size_average") = true,
                  py::arg("ignore_index") = -100,
                  py::arg("reduce") = true,
                  py::arg("reduction") = "mean");
    }

    //
    // interpolate
    //
    THPNestedTensor interpolate(const THPNestedTensor& input,
                                c10::optional<std::vector<int64_t>> size,
                                c10::optional<std::vector<double>> scale_factor,
                                c10::optional<std::string> mode,
                                c10::optional<bool> align_corners) {
                                //bool recompute_scale_factor) { 
        TensorNode input_structure = input.data().contiguous().get_structure();
        TensorNode res = map([&] (at::Tensor input_tensor) {
            F::InterpolateFuncOptions::mode_t int_mode;
            if (mode.value() == "nearest" || mode.value() == "none") {
                int_mode = torch::kNearest;
            } else if (mode.value()== "trilinear") {
                int_mode = torch::kTrilinear;
            } else {
                std::stringstream error;
                error << "Unexpected mode for interpolate: " << mode.value() << std::endl;
                throw error;
            }

            auto options = F::InterpolateFuncOptions().mode(int_mode);

            if (scale_factor.has_value()) {
                options.scale_factor() = scale_factor.value();
            }

            if (size.has_value()) {
                if (size.value().size() == 2) {
                    options.size() = size.value();
                } else {
                    options.size() = std::vector<int64_t>({size.value()[0], size.value()[0]});
                }
            }

            // size or scale factor have to be defined
            if (!size.has_value() && !scale_factor.has_value()) {
                std::vector<int64_t> sizes;
                sizes.push_back(input_tensor.unsqueeze(0).size(2));
                sizes.push_back(input_tensor.unsqueeze(0).size(2));
                options.size() = sizes;
            }

            return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
        }, input_structure);

        return THPNestedTensor(NestedTensor(std::move(res)));
    }

    void add_interpolate(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), 
                  interpolate,
                  py::arg("input"),
                  py::arg("size") = nullptr,
                  py::arg("scale_factor") = nullptr,
                  py::arg("mode") = "nearest",
                  py::arg("align_corners") = false);
                  //py::arg("recompute_scale_factor") = false);
    }

    // from python side, size can come as None, a single integer or a tuple. 
    // this wrapper covers single integer case.
    THPNestedTensor interpolate_single_size(const THPNestedTensor& input,
                                            int64_t size,
                                            c10::optional<std::vector<double>> scale_factor,
                                            c10::optional<std::string> mode,
                                            c10::optional<bool> align_corners) {
        return interpolate(input, std::vector<int64_t>({size}), scale_factor, mode, align_corners);
    }

    void add_interpolate_single_size(
        pybind11::module m,
        pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
        std::string name) {
            m.def(name.c_str(), 
                  interpolate_single_size,
                  py::arg("input"),
                  py::arg("size"),
                  py::arg("scale_factor") = nullptr,
                  py::arg("mode") = "nearest",
                  py::arg("align_corners") = false);
                  //py::arg("recompute_scale_factor") = false);
    }
}
}