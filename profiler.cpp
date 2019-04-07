

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

void StartProfilingPytorch(Torch_PredictorContext pred, const char *name, const char *metadata) {
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

void EndProfilingPytorch(Torch_PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void EnableProfilingPytorch(Torch_PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->profile_enabled_ = true;
}

void DisableProfilingPytorch(Torch_PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
  predictor->profile_enabled_ = false;
}

char *ReadProfilePytorch(Torch_PredictorContext pred) {
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
    LOG(ERROR) << "exception: catch all [ " << ex.what() << "]"
               << "\n";
    return nullptr;
  }
}
