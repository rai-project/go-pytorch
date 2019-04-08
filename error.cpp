

#include "error.hpp"
#include "predictor.hpp"

#ifdef ENABLE_BACKWARD
#include "backward.hpp"
namespace backward {

backward::SignalHandling sh;

}  // namespace backward

#endif  // ENABLE_BACKWARD

Torch_Error Torch_GlobalError{.message = nullptr};

char Torch_HasError() { return Torch_GlobalError.message == nullptr ? 0 : 1; }

const char* Torch_GetErrorString() {
  if (!Torch_HasError()) {
    return nullptr;
  }
  return Torch_GlobalError.message;
}

void Torch_ResetError() {
  if (Torch_HasError()) {
    free(Torch_GlobalError.message);
    Torch_GlobalError.message = nullptr;
  }
}