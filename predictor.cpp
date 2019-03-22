#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
//#include <ATen/core/DefaultTensorOptions.h>

#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

using namespace torch;
using std::string;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

/*
  Predictor class takes in one module file (exported using torch JIT compiler)
  , batch size and device mode for inference
*/
class Predictor {
  public:
    Predictor(const string &model_file, int batch, torch::DeviceType mode);
    void Predict(float* inputData);

    std::shared_ptr<torch::jit::script::Module> net_;
    int width_, height_, channels_;
    int batch_;
    int pred_len_;
    torch::DeviceType mode_{torch::kCPU};
    profile *prof_{nullptr};
    bool profile_enabled_{false};
    at::Tensor result_;
};

Predictor::Predictor(const string &model_file, int batch, torch::DeviceType mode) {
  
  // Load the network
  net_ = torch::jit::load(model_file);
  assert(net_ != nullptr);
  mode_ = mode;

  // Input shape hard coded for now due to lack of layer shape accessor API
  // Preferred alternate: pass input layer shape as an input 
  width_ = 224;
  height_ = 224;
  channels_ = 3;
  batch_ = batch;

  CHECK(channels_ == 3 || channels_ == 1) << "Input layer should have 1 or 3 channels.";

}

void Predictor::Predict(float* inputData) {

  std::vector<int64_t> sizes = {1, 3, width_, height_};
  at::TensorOptions options(at::kFloat);
  at::Tensor tensor_image = torch::from_blob(inputData, at::IntList(sizes), options);

  std::vector<torch::jit::IValue> inputs;
  if(mode_ == torch::kCUDA) {
    net_->to(at::kCUDA);
    at::Tensor tensor_image_cuda = tensor_image.to(at::kCUDA);
    inputs.emplace_back(tensor_image_cuda);
    result_ = net_->forward(inputs).toTensor();
  }else {
    inputs.emplace_back(tensor_image);
    result_ = net_->forward(inputs).toTensor();
  }
  result_ = result_.to(at::kCPU);

}

PredictorContext NewPytorch(char *model_file, int batch, int mode) {
  
  try {
    torch::DeviceType mode_temp{at::kCPU};
    if (mode == 1) mode_temp = at::kCUDA;
    const auto ctx = new Predictor(model_file, batch, (torch::DeviceType)mode_temp);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }

}

void SetModePytorch(int mode) {
  if(mode == 1) torch::Device device(torch::kCUDA);
}

void InitPytorch() {}

void PredictPytorch(PredictorContext pred, float* inputData) {
  
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;

}

const float*GetPredictionsPytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }

  return predictor->result_.data<float>();

}

void DeletePytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
    delete predictor->prof_;
    predictor->prof_ = nullptr;
  }
  delete predictor;

}

int GetWidthPytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->width_;

}

int GetHeightPytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->height_;

}

int GetChannelsPytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->channels_;

}

int GetPredLenPytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  predictor->pred_len_ = predictor->result_.size(1);
  return predictor->pred_len_;

}

