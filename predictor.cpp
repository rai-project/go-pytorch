

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

extern Torch_IValue Torch_ConvertIValueToTorchIValue(torch::IValue value);

class Predictor {
 public:
  Predictor(const string &model_file, int batch, DeviceKind device);
  void Predict(float *inputData);

  std::shared_ptr<torch::jit::script::Module> net_;
  std::vector<torch::jit::IValue> inputs_;
  torch::jit::IValue output_;
  torch::DeviceType mode_{torch::kCPU};
  profile *prof_{nullptr};
  std::stringstream ss_;
  std::string profile_filename_{"profile.trace"};
  bool profile_enabled_{false};
  at::Tensor result_;
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
  HANDLE_TH_ERRORS
  DeviceKind device_temp{CPU_DEVICE_KIND};
  if (mode == 1) device_temp = CUDA_DEVICE_KIND;
  const auto ctx = new Predictor(model_file, batch, (DeviceKind)device_temp);
  return (void *)ctx;
  END_HANDLE_TH_ERRORS(error, nullptr);
}

void Torch_PredictorSetMode(Torch_DeviceKind mode) { mode_ = mode; }

void InitPytorch() {}

void Torch_PredictorRun(PredictorContext pred, float *inputData) {
  HANDLE_TH_ERRORS
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  END_HANDLE_TH_ERRORS(error, );
}

const int Torch_PredictorNumOutputs(PredictorContext pred) {
  HANDLE_TH_ERRORS
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->result_tensors_.size();
  END_HANDLE_TH_ERRORS(error, 0);
}

Torch_IValue Torch_PredictorGetOutput(PredictorContext pred) {
  HANDLE_TH_ERRORS
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }

  output_ = output_.to(at::kCPU);
  return Torch_ConvertIValueToTorchIValue(output_);

  END_HANDLE_TH_ERRORS(error, Torch_IValue{});
}

void Torch_PredictorDelete(PredictorContext pred) {
  HANDLE_TH_ERRORS
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
  END_HANDLE_TH_ERRORS(error, );
}
