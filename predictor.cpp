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
#include <ATen/core/DefaultTensorOptions.h>

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
    // void Predict(float* inputData);
    void Predict(int64_t* inputData);

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
  /* Load the network. */
  // In pytorch, a loaded module in c++ is given 
  // type torch::jit::script::Module as it has been
  // ported from python/c++ via pytorch's JIT compiler
  net_ = torch::jit::load(model_file);
  assert(net_ != nullptr);
  mode_ = mode;

  // TODO should fetch width and height from model
  //const torch::detail::OrderedDict<std::string, torch::jit::script::NamedModule>& net_module_dict = net_->get_modules();
  //size_t net_module_dict_size = net_module_dict.size();
  //CHECK((int)net_module_dict_size == 1) << "Number of modules - " << (int)net_module_dict_size;  
  //const torch::jit::script::NamedModule& temp = net_module_dict.get("fc1"); 
  //temp.module->get_method("fc1_script");  

  width_ = 3;
  height_ = 1;
  channels_ = 1;
  batch_ = batch;

  CHECK(channels_ == 3 || channels_ == 1) << "Input layer should have 1 or 3 channels.";

}

/*
// Predict for Alexnet
void Predictor::Predict(float* inputData) {

  std::vector<int64_t> sizes = {1, 3, width_, height_};
  at::TensorOptions options(at::kFloat);
  at::Tensor tensor_image = torch::from_blob(inputData, at::IntList(sizes), options);

  std::vector<torch::jit::IValue> inputs;

  // check if mode is set to GPU
  if(mode_ == torch::kCUDA) {
    // port model to GPU
    net_->to(at::kCUDA);
    // port input to GPU
    at::Tensor tensor_image_cuda = tensor_image.to(at::kCUDA);
    // emplace IValue input
    inputs.emplace_back(tensor_image_cuda);
    // execute model
    result_ = net_->forward(inputs).toTensor();
  }else {
    // emplace IValue input
    inputs.emplace_back(tensor_image);
    // execute model
    result_ = net_->forward(inputs).toTensor();
  }
  
  // port output back to CPU
  result_ = result_.to(at::kCPU);

}
*/

// changing Predict for NCF model
void Predictor::Predict(int64_t* inputData) {

  std::vector<int64_t> sizes = {1};
  at::TensorOptions options(at::kLong);
  at::Tensor tensor_inp0 = torch::from_blob(inputData, at::IntList(sizes), options);
  at::Tensor tensor_inp1 = torch::from_blob(inputData+1, at::IntList(sizes), options);
  at::Tensor tensor_inp2 = torch::from_blob(inputData+2, at::IntList(sizes), options);

  std::vector<torch::jit::IValue> inputs;

  // check if mode is set to GPU
  if(mode_ == torch::kCUDA) {
    // port model to GPU
    net_->to(at::kCUDA);
    // port input to GPU
    at::Tensor tensor_inp0_cuda = tensor_inp0.to(at::kCUDA);
    at::Tensor tensor_inp1_cuda = tensor_inp1.to(at::kCUDA);
    at::Tensor tensor_inp2_cuda = tensor_inp2.to(at::kCUDA);
    // emplace IValue input
    inputs.emplace_back(tensor_inp0_cuda);
    inputs.emplace_back(tensor_inp1_cuda);
    inputs.emplace_back(tensor_inp2_cuda);
    // execute model
    result_ = net_->forward(inputs).toTensor();
  }else {
    // emplace IValue input
    inputs.emplace_back(tensor_inp0);
    inputs.emplace_back(tensor_inp1);
    inputs.emplace_back(tensor_inp2);
    // execute model
    result_ = net_->forward(inputs).toTensor();
  }
  
  // port output back to CPU
  result_ = result_.to(at::kCPU);

}



PredictorContext NewPytorch(char *model_file, int batch,
                          int mode) {
  try {
    torch::DeviceType mode_temp{at::kCPU};
    if (mode == 1) {
      mode_temp = at::kCUDA;
    }
    const auto ctx = new Predictor(model_file, batch,
                                   (torch::DeviceType)mode_temp);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }

}

void SetModePytorch(int mode) {
  if(mode == 1) {
    // TODO set device here ?
    torch::Device device(torch::kCUDA);
  }
}

void InitPytorch() {}

/*
// Predict for Alexnet
void PredictPytorch(PredictorContext pred, float* inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;
}
*/

// Predict for NCF
/*
 Assuming that the float* input data consists of three integers as float
 which are item id, user id, and a boolean flag
 converting float to int in order to match the predictor
*/
void PredictPytorch(PredictorContext pred, float* inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  int64_t* IntegerinputData = new int64_t[3];
  IntegerinputData[0] = int64_t(inputData[0]);
  IntegerinputData[1] = int64_t(inputData[1]);
  IntegerinputData[2] = int64_t(inputData[2]);
  predictor->Predict(IntegerinputData);
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

