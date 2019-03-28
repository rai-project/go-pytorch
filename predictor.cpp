//#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>
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

  template <typename T>
  void AddInput(int ii, T *inputData);

  void Predict();

  std::shared_ptr<torch::jit::script::Module> net_;
  std::vector<std::vector<int>> shapes_{};

  int batch_;

  int pred_len_;
  torch::DeviceType mode_{torch::kCPU};
  profile *prof_{nullptr};
  bool profile_enabled_{false};
  at::Tensor result_;

  std::vector<torch::jit::IValue> inputs_{};
};

Predictor::Predictor(const string &model_file, int batch,
                     torch::DeviceType mode) {
  /* Load the network. */
  // In pytorch, a loaded module in c++ is given
  // type torch::jit::script::Module as it has been
  // ported from python/c++ via pytorch's JIT compiler
  net_ = torch::jit::load(model_file);
  assert(net_ != nullptr);
  mode_ = mode;
  batch_ = batch;
}

template <typename T>
void Predictor::AddInput(int index, T *inputData, int *sizes, int sizeLen) {
  std::vector<int> shape(sizeLen, sizes);
  shapes_.emplace_back(shape);

  at::TensorOptions options;
#define SET_TYPE(ty, top, _) \
  if (std::is_same<T, ty>::value) options = at::ty;
  AT_FORALL_SCALAR_TYPES(SET_TYPE);
#undef SET_TYPE

  at::Tensor tensor_image =
      torch::from_blob(inputData, at::IntList(shape), options);

  // check if mode is set to GPU
  if (mode_ == torch::kCUDA) {
    // port model to GPU
    net_->to(at::kCUDA);
    at::Tensor tensor_image_cuda = tensor_image.to(at::kCUDA);
    inputs.emplace_back(tensor_image_cuda);
  } else {
    // emplace IValue input
    inputs.emplace_back(tensor_image);
  }
}

void Predictor::Predict() {
  result_ = net_->forward(inputs).toTensor();
  // port output back to CPU
  result_ = result_.to(at::kCPU);
}

PredictorContext NewPytorch(char *model_file, int batch, int mode) {
  try {
    torch::DeviceType mode_temp{at::kCPU};
    if (mode == 1) {
      mode_temp = at::kCUDA;
    }
    const auto ctx =
        new Predictor(model_file, batch, (torch::DeviceType)mode_temp);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void SetModePytorch(int mode) {
  if (mode == 1) {
    // TODO set device here ?
    torch::Device device(torch::kCUDA);
  }
}

void InitPytorch() {}

void PredictPytorch(PredictorContext pred, float *inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;
}

const float *GetPredictionsPytorch(PredictorContext pred) {
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
