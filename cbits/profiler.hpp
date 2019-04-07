
#pragma once

#include "timer.h"
#include "timer.impl.hpp"
class StartProfile {
 public:
  explicit StartProfile(profile *prof, const std::shared_ptr<torch::jit::script::Module> &net)
      : prof_(prof), net_(net) {}
  virtual ~StartProfile() {}

 protected:
  virtual void run() final {
    if (prof_ == nullptr || net_ == nullptr) {
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
    if (prof_ == nullptr) {
      return;
    }
  }

 private:
  profile *prof_{nullptr};
};
