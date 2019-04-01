// TODO need to add a check - if in docker, comment it
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
#include <../../autograd/profiler.h>

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
    explicit StartProfile(profile *prof, const std::shared_ptr<torch::jit::script::Module> &net) : prof_(prof), net_(net) {}
    virtual ~StartProfile() {}

  protected:
    virtual void run() final {
      if(prof_ == nullptr || net_ == nullptr) {
        return;
      }
    }

  private:
    profile *prof_{nullptr};
    std::shared_ptr<torch::jit::script::Module> net_{nullptr};
};

class EndProfile {
  public:
    explicit EndProfile(profile *prof) : prof_(prof) {}
    virtual ~EndProfile() {}

  protected:
    virtual void run() final {
      if(prof_ == nullptr) {
        return;
      }

    }
	
  private:
    profile *prof_{nullptr};
};

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
    std::stringstream ss_;
    std::string filename{"profile.trace"};
    bool profile_enabled_{false};
    at::Tensor result_;
};

Predictor::Predictor(const string &model_file, int batch, torch::DeviceType mode) {
  
  // Load the network
  net_ = torch::jit::load(model_file);
  assert(net_ != nullptr);
  mode_ = mode;
  batch_ = batch;

}

void Predictor::Predict(float* inputData) {

  std::vector<int64_t> sizes = {batch_, channels_, width_, height_};
  at::TensorOptions options(at::kFloat);
  at::Tensor tensor_image = torch::from_blob(inputData, at::IntList(sizes), options);

  std::vector<torch::jit::IValue> inputs;
  if(mode_ == torch::kCUDA) {
    net_->to(at::kCUDA);
    at::Tensor tensor_image_cuda = tensor_image.to(at::kCUDA);
    inputs.emplace_back(tensor_image_cuda);
    if (profile_enabled_) {
      {
        //autograd::profiler::enableProfiler(autograd::profiler::ProfilerState::CUDA);
        autograd::profiler::RecordProfile guard(filename);
        result_ = net_->forward(inputs).toTensor();
      }
    } else {
      result_ = net_->forward(inputs).toTensor();
    }
  }else {
    inputs.emplace_back(tensor_image);
    if (profile_enabled_) {
      {
        //autograd::profiler::enableProfiler(autograd::profiler::ProfilerState::CPU);
        autograd::profiler::RecordProfile guard(filename);
        result_ = net_->forward(inputs).toTensor();
      }
    } else {
      result_ = net_->forward(inputs).toTensor();
    }
  }
  result_ = result_.to(at::kCPU);

}

PredictorContext NewPytorch(char *model_file, int batch, int mode) {
  
  try {
    torch::DeviceType mode_temp{torch::kCPU};
    if (mode == 1) mode_temp = torch::kCUDA;
    const auto ctx = new Predictor(model_file, batch, (torch::DeviceType)mode_temp);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }

}

void SetModePytorch(int mode) {
  if(mode == 1) {
    //mode_ = torch::kCUDA;
  }
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

void SetDimensionsPytorch(PredictorContext pred, int channels, int height, int width, int batch) {

  auto predictor = (Predictor *)pred;
  if(predictor == nullptr) {
    return;
  }
  predictor->channels_ = channels;
  predictor->height_ = height;
  predictor->width_ = width;
  predictor->batch_ = batch;

}

int GetPredLenPytorch(PredictorContext pred) {

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  predictor->pred_len_ = predictor->result_.size(1);
  return predictor->pred_len_;

}

void StartProfilingPytorch(PredictorContext pred, const char *name, const char *metadata) {
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

void EnableProfilingPytorch(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->profile_enabled_ = true;

}

void DisableProfilingPytorch(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
  predictor->profile_enabled_ = false;

}

char *ReadProfilePytorch(PredictorContext pred) {
  try {  
    auto predictor = (Predictor *)pred;
    if (predictor == nullptr) {
      return strdup("");
    }
    if (predictor->prof_ == nullptr) {
      return strdup("");
    }
    const auto prof_output = predictor->ss_.str().c_str();
    return strdup(prof_output);
  } catch (std::exception &ex) {
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]" << "\n";
    return nullptr;
  }
}
