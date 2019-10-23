

#include "error.hpp"
#include "predictor.hpp"
#include "profiler.hpp"
#include "timer.h"
#include "timer.impl.hpp"

#include <algorithm>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

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
  Predictor(const string &model_file, Torch_DeviceKind device);
  void Predict(Torch_TensorContext *cInputs, int inputLength);
  torch::jit::script::Module net_;
  torch::IValue output_;
  torch::DeviceType mode_{torch::kCPU};

  std::stringstream ss_;
  profile *prof_{nullptr};
  std::string profile_filename_{"profile.trace"};
  bool profile_enabled_{false};
};

Predictor::Predictor(const string &model_file, Torch_DeviceKind device) {
  // Load the network
  net_ = torch::jit::load(model_file);
  if (device == CUDA_DEVICE_KIND) mode_ = torch::kCUDA;

  if (mode_ == torch::kCUDA) {
    net_.to(at::kCUDA);
  }
#ifdef PROFILING_ENABLED
  profile_enabled_ = true;
#endif
}

void Predictor::Predict(Torch_TensorContext *cInputs, int inputLength) {
  std::vector<torch::jit::IValue> inputs{};

  for (int ii = 0; ii < inputLength; ii++) {
    at::Tensor tensor = reinterpret_cast<Torch_Tensor *>(cInputs[ii])->tensor;

    std::cout << "tensor dim = " << tensor.dim() << " size = ";
    for (auto sz : tensor.sizes()) {
      std::cout << sz << ", ";
    }
    std::cout << "\n";
    inputs.emplace_back(tensor);
  }

  if ((profile_enabled_ == true) && (mode_ == torch::kCPU)) {
    autograd::profiler::RecordProfile guard(profile_filename_);
    output_ = net_.forward(inputs);
    return;
  }
 
  if ((profile_enabled_ == true) && (mode_ == torch::kCUDA)) {
    autograd::profiler::enableProfiler(autograd::profiler::ProfilerConfig(autograd::profiler::ProfilerState::CUDA, true));
    output_ = net_.forward(inputs);
    // TODO: should we synchronize CUDA execution ?
    autograd::profiler::thread_event_lists event_lists = autograd::profiler::disableProfiler();
    std::vector<autograd::profiler::Event*> events;
    for(auto& l: event_lists) {
      for(auto& e: l) {
        events.push_back(&e);
        // DEBUG
        //std::cout << "Event kind: " << e.name() << std::endl;
      }
    }
    std::ofstream* file_ = new std::ofstream(profile_filename_);
    std::ostream& out_ = *(file_);
    // DEBUG
    //std::cout << "Searching for start event..." << std::endl;
    autograd::profiler::Event* start = nullptr;
    for (autograd::profiler::Event* e : events) {
      if(0 == strcmp(e->name(), "__start_profile")) {
        start = e;
        // DEBUG
        //std::cout << "Found a start event in CUDA Profile!" << std::endl;
        break;
      }
    }
    //autograd::profiler::TORCH_CHECK(start, "could not find start event");
    std::vector<autograd::profiler::Event*> stack;
    out_ << "[\n";
    bool first = true;
    for (autograd::profiler::Event* e: events) {
      if (e->kind() == "push") {
        stack.push_back(e);
      } else if (e->kind() == "pop") {
        if(!first) {
          out_ << ",\n";
        }
        first = false;
        autograd::profiler::Event* e_start = stack.back();
        stack.pop_back();
        jit::TemplateEnv env;
        env.s("name", e_start->name());
        env.d("ts", start->cpu_elapsed_us(*e_start));
        env.d("dur", e_start->cpu_elapsed_us(*e));
        env.d("tid", e_start->thread_id());
        static jit::CodeTemplate event_template(R"(
        {
          "name": "${name}",
          "ph": "X",
          "ts": ${ts},
          "dur": ${dur},
          "tid": ${tid},
          "pid": "CUDA Functions",
          "args": {}
        })");
        out_ << event_template.format(env);
      }
    }
    out_ << "]\n"; 
    if(file_)
      file_->close();

    return;
  }

  output_ = net_.forward(inputs);
}

Torch_PredictorContext Torch_NewPredictor(const char *model_file, Torch_DeviceKind mode) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  const auto ctx = new Predictor(model_file, mode);
  return (Torch_PredictorContext)ctx;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, (Torch_PredictorContext)0);
}

void InitPytorch() {}

void Torch_PredictorRun(Torch_PredictorContext pred, Torch_TensorContext *cInputs, int inputLength) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(cInputs, inputLength);
  END_HANDLE_TH_ERRORS(Torch_GlobalError, );
}

int Torch_PredictorNumOutputs(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  if (predictor->output_.isTensor()) {
    return 1;
  }
  if (predictor->output_.isTuple()) {
    return predictor->output_.toTuple()->elements().size();
  }

  return 0;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, 0);
}

Torch_IValue Torch_PredictorGetOutput(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return Torch_IValue{};
  }

  return Torch_ConvertIValueToTorchIValue(predictor->output_);

  END_HANDLE_TH_ERRORS(Torch_GlobalError, Torch_IValue{});
}

void Torch_PredictorDelete(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
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
  END_HANDLE_TH_ERRORS(Torch_GlobalError, );
}

void Torch_ProfilingStart(Torch_PredictorContext pred, const char *name, const char *metadata) {
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

void Torch_ProfilingEnd(Torch_PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void Torch_ProfilingEnable(Torch_PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->profile_enabled_ = true;
}

void Torch_ProfilingDisable(Torch_PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
  predictor->profile_enabled_ = false;
}

char *Torch_ProfilingRead(Torch_PredictorContext pred) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return strdup("");
  }
  if (predictor->prof_ == nullptr) {
    return strdup("");
  }
  const auto prof_output = predictor->ss_.str().c_str();
  return strdup(prof_output);

  END_HANDLE_TH_ERRORS(Torch_GlobalError, (char *)0);
}
