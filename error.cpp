

#include "error.hpp"
#include "predictor.hpp"

Torch_Error Torch_GlobalError{.message = nullptr};

char Torch_HasError() { return Torch_GlobalError.message != nullptr; }

const char* Torch_GetErrorString() {
  if (!Torch_HasError()) {
    return "success";
  }
  return Torch_GlobalError.message;
}

void Torch_ResetError() {
  if (Torch_HasError()) {
    free(Torch_GlobalError.message);
    Torch_GlobalError.message = nullptr;
  }
}