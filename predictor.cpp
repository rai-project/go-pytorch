

#include <algorithm>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "error.hpp"
#include "predictor.hpp"
#include "profiler.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

using namespace torch;
using std::string;

class Predictor {
 public:
  Predictor(const string &model_file, int batch, DeviceKind device);
  void Predict(float *inputData);

  std::shared_ptr<torch::jit::script::Module> net_;
  std::vector<torch::jit::IValue> inputs_;
  torch::DeviceType mode_{torch::kCPU};
  profile *prof_{nullptr};
  std::stringstream ss_;
  std::string profile_filename_{"profile.trace"};
  bool profile_enabled_{false};
  at::Tensor result_;
  std::vector<at::Tensor> result_tensors;
};

Predictor::Predictor(const string &model_file, int batch, DeviceKind device) {
  // Load the network
  net_ = torch::jit::load(model_file);
  assert(net_ != nullptr);
  if (device == CUDA_DEVICE_KIND) mode_ = torch::kCUDA;
}

void Predictor::AddInput() {
  if (mode_ == torch::kCUDA) {
    net_->to(at::kCUDA);
    for (auto input : inputs_) {
      input.to(at::kCUDA);
    }
  }
}

void Predictor::Predict() {
#ifdef PROFILING_ENABLED
  autograd::profiler::RecordProfile profile_recorder;
#endif  // PROFILING_ENABLED

#ifdef PROFILING_ENABLED
  if (profile_enabled_) {
    profile_recorder = autograd::profiler::RecordProfile(profile_filename_);
    result_ = net_->forward(inputs_);
    return;
  }
#endif  // PROFILING_ENABLED
  result_ = net_->forward(inputs_);
}

PredictorContext Torch_NewPredictor(char *model_file, int batch, int mode) {
  try {
    DeviceKind device_temp{CPU_DEVICE_KIND};
    if (mode == 1) device_temp = CUDA_DEVICE_KIND;
    const auto ctx = new Predictor(model_file, batch, (DeviceKind)device_temp);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void Torch_PredictorSetMode(int mode) {
  if (mode == 1) {
    // mode_ = torch::kCUDA;
  }
}

void InitPytorch() {}

void Torch_PredictorRun(PredictorContext pred, float *inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;
}

const int Torch_PredictorNumOutputs(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->result_tensors.size();
}

// returns int array of individual tensor sizes
const int *Torch_PredictorOutput(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  int num_tensors = predictor->result_tensors.size();
  std::cout << "No of tensors - " << num_tensors << std::endl;
  std::vector<int> size_of_tensors;
  size_of_tensors.reserve(num_tensors);
  for (int i = 0; i < num_tensors; i++) {
    int num_dims = predictor->result_tensors[i].sizes().size();
    int length = 1;
    for (int j = 0; j < num_dims; j++) length *= predictor->result_tensors[i].sizes().data()[j];
    size_of_tensors.emplace_back(length);
  }
  return size_of_tensors.data();
}

const float *GetPredictionsPytorch(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  std::cout << "I am here - 1!" << std::endl;
  const int *size_of_tensors = Torch_PredictorOutput(pred);
  for (int i = 0; i < 2; i++) std::cout << "size of tensor[" << i << "] - " << size_of_tensors[i] << std::endl;
  std::cout << "I am here - 2!" << std::endl;
  int total_size_of_tensors = 0;
  for (int i = 0; i < predictor->result_tensors.size(); i++) {
    std::cout << "size of tensor " << i << " : " << size_of_tensors[i];
    total_size_of_tensors += size_of_tensors[i];
  }
  float *combined = new float[total_size_of_tensors];
  for (size_t i = 0; i < predictor->result_tensors.size(); i++) {
    std::copy(predictor->result_tensors[i].data<float>(),
              predictor->result_tensors[i].data<float>() + size_of_tensors[i], combined);
  }
  return combined;
}

void Torch_PredictorDelete(PredictorContext pred) {
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

int GetPredLenPytorch(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  // predictor->pred_len_ = predictor->result_.size(1);
  return predictor->pred_len_;
}
