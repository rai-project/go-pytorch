// TODO need to add a check - if in docker, comment it
#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <../../autograd/profiler.h>
#include <torch/script.h>

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

class StartProfile {
 public:
  explicit StartProfile(profile *prof,
                        const std::shared_ptr<torch::jit::script::Module> &net)
      : prof_(prof), net_(net) {}
  virtual ~StartProfile() {}

 protected:
  virtual void run() final {
    if (prof_ == nullptr || net_ == nullptr) {
      return;
    }
    // TODO  start autograd profiler
  }

 private:
  profile *prof_{nullptr};
  const shared_ptr<torch::jit::script::Module> net_{nullptr};
};

class EndProfile {
 public:
  explicit EndProfile(profile *prof) : prof_(prof) {}
  virtual ~EndProfile() {}

 protected:
  virtual void run() final {
    if (prof_ == nullptr) {
      return;
    }
    // TODO end autograd profiler
  }

 private:
  profile *prof_{nullptr};
};

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

  // TODO should fetch width and height from model
  // const torch::detail::OrderedDict<std::string,
  // torch::jit::script::NamedModule>& net_module_dict = net_->get_modules();
  // size_t net_module_dict_size = net_module_dict.size();
  // CHECK((int)net_module_dict_size == 1) << "Number of modules - " <<
  // (int)net_module_dict_size; const torch::jit::script::NamedModule& temp =
  // net_module_dict.get("fc1"); temp.module->get_method("fc1_script");

  // Input shape hard coded for now due to absence of layer shape through API
  // TODO Preferred alternative: pass input layer shape as an input
  width_ = 224;
  height_ = 224;
  channels_ = 3;
  batch_ = batch;
}

template <typename T>
void Predictor::AddInput(int index, T *inputData, int *sizes, int sizeLen) {
  std::vector<int> shape(sizeLen, sizes);
  shapes_.emplace_back(shape);

  //   at::TensorOptions options;
  // #define SET_TYPE(ty, top, _) \
//   if (std::is_same<T, ty>::value) options = at::ty;
  //   AT_FORALL_SCALAR_TYPES(SET_TYPE);
  // #undef SET_TYPE

  auto options = at::dtype<T>();

  at::Tensor tensor_image =
      torch::from_blob(inputData, at::IntList(shape), options);

  StartProfile *start_profile = nullptr;
  EndProfile *end_profile = nullptr;
  if (prof_ != nullptr && profile_enabled_ == false) {
    start_profile = new StartProfile(prof_, net_);
    end_profile = new EndProfile(prof_);
    profile_enabled_ = true;
  }

  std::vector<torch::jit::IValue> inputs;
  if (mode_ == torch::kCUDA) {
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

void StartProfilingPytorch(PredictorContext pred, const char *name,
                           const char *metadata) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  if (predictor->prof_ == nullptr) {
    predictor->prof_ = new profile(name, metadata);
  } else {
    predictor->prof_->reset();
  }
}

void EndProfilingPytorch(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void DisableProfilingPytorch(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
}

char *ReadProfilePyTorch(PredictorContext pred) {
  auto predictor = (Predictor * pred);
  if (predictor == nullptr) {
    return strdup("");
  }
  if (predictor->prof_ == nullptr) {
    return strdup("");
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}
