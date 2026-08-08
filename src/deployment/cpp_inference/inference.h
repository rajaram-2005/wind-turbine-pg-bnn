/**
 * @file inference.h
 * @brief ONNX Runtime inference engine for the physics-guided BNN on edge devices.
 *
 * Loads the ONNX model exported by src/deployment/export_onnx.py and runs
 * deterministic (posterior-mean) inference, returning both the predictive
 * mean and the aleatoric variance for each output target.
 */

#ifndef PG_BNN_INFERENCE_H
#define PG_BNN_INFERENCE_H

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace pgbnn {

/// Result of one inference call: per-target mean and variance.
struct InferenceResult {
    std::vector<float> mean;      ///< Predictive mean per output target.
    std::vector<float> variance;  ///< Aleatoric variance per output target.
};

/**
 * @brief Thin RAII wrapper around an ONNX Runtime session for the PG-BNN.
 *
 * Usage:
 * @code
 *   pgbnn::OnnxInferenceEngine engine("pg_bnn.onnx");
 *   std::vector<float> features = {7.4f, 1520.f, 62.5f, 2.4f, 33.f, 81.f};
 *   auto result = engine.Run(features);
 *   float rul_mean = result.mean[0];
 *   float rul_std  = std::sqrt(result.variance[0]);
 * @endcode
 */
class OnnxInferenceEngine {
public:
    /**
     * @brief Construct the engine and load the ONNX model.
     * @param model_path Filesystem path to the exported .onnx model.
     * @param num_threads Intra-op thread count (edge devices: keep small).
     * @throws Ort::Exception if the model cannot be loaded.
     */
    explicit OnnxInferenceEngine(const std::string& model_path, int num_threads = 1);

    ~OnnxInferenceEngine();
    OnnxInferenceEngine(const OnnxInferenceEngine&) = delete;
    OnnxInferenceEngine& operator=(const OnnxInferenceEngine&) = delete;

    /**
     * @brief Run inference on a single feature vector.
     * @param features Input SCADA feature vector (length must equal InputDim()).
     * @return Mean and variance for every output target.
     * @throws std::invalid_argument if the feature length mismatches the model.
     */
    InferenceResult Run(const std::vector<float>& features) const;

    /**
     * @brief Run batched inference.
     * @param batch Row-major batch of shape (batch_size x InputDim()).
     * @param batch_size Number of rows in the batch.
     * @return One InferenceResult per row.
     */
    std::vector<InferenceResult> RunBatch(const std::vector<float>& batch,
                                          size_t batch_size) const;

    /// @return Number of input features the model expects.
    size_t InputDim() const;

    /// @return Number of output targets per sample.
    size_t OutputDim() const;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    size_t input_dim_ = 0;
    size_t output_dim_ = 0;
};

}  // namespace pgbnn

#endif  // PG_BNN_INFERENCE_H
