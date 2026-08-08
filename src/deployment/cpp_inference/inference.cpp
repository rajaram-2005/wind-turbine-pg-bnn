/**
 * @file inference.cpp
 * @brief Implementation of the ONNX Runtime edge-inference engine.
 */

#include "inference.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pgbnn {

OnnxInferenceEngine::OnnxInferenceEngine(const std::string& model_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "pg-bnn-edge") {
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;

    // Input: "scada_features" with shape (batch, input_dim)
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name.get();
    auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();
    input_dim_ = input_shape.size() >= 2 && input_shape[1] > 0
                     ? static_cast<size_t>(input_shape[1])
                     : 0;

    // Outputs: "mean" and "variance"
    const size_t n_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < n_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }
    auto output_info = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    output_dim_ = output_shape.size() >= 2 && output_shape[1] > 0
                      ? static_cast<size_t>(output_shape[1])
                      : 1;
}

OnnxInferenceEngine::~OnnxInferenceEngine() = default;

size_t OnnxInferenceEngine::InputDim() const { return input_dim_; }

size_t OnnxInferenceEngine::OutputDim() const { return output_dim_; }

InferenceResult OnnxInferenceEngine::Run(const std::vector<float>& features) const {
    if (input_dim_ != 0 && features.size() != input_dim_) {
        throw std::invalid_argument("feature length " + std::to_string(features.size()) +
                                    " != model input dim " + std::to_string(input_dim_));
    }
    auto batch_results = RunBatch(features, /*batch_size=*/1);
    return batch_results.front();
}

std::vector<InferenceResult> OnnxInferenceEngine::RunBatch(const std::vector<float>& batch,
                                                           size_t batch_size) const {
    const size_t dim = input_dim_ != 0 ? input_dim_ : batch.size() / batch_size;
    if (batch.size() != batch_size * dim) {
        throw std::invalid_argument("batch buffer size does not match batch_size x input_dim");
    }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size),
                                              static_cast<int64_t>(dim)};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(batch.data()), batch.size(), input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    std::vector<const char*> output_names;
    output_names.reserve(output_names_.size());
    for (const auto& n : output_names_) output_names.push_back(n.c_str());

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                 output_names.data(), output_names.size());

    const float* mean_data = outputs[0].GetTensorData<float>();
    const float* var_data =
        outputs.size() > 1 ? outputs[1].GetTensorData<float>() : nullptr;
    const size_t out_dim =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape().back() > 0
            ? static_cast<size_t>(outputs[0].GetTensorTypeAndShapeInfo().GetShape().back())
            : output_dim_;

    std::vector<InferenceResult> results(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        results[b].mean.assign(mean_data + b * out_dim, mean_data + (b + 1) * out_dim);
        if (var_data != nullptr) {
            results[b].variance.assign(var_data + b * out_dim, var_data + (b + 1) * out_dim);
        } else {
            results[b].variance.assign(out_dim, 0.0f);
        }
    }
    return results;
}

}  // namespace pgbnn
