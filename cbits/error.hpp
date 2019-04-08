#pragma once

#include <string.h>

#define HANDLE_TH_ERRORS(errVar)     \
  try {                              \
    if (errVar.message != nullptr) { \
      free(errVar.message);          \
    }                                \
    errVar.message = 0;
#define END_HANDLE_TH_ERRORS(errVar, retVal)              \
  }                                                       \
  catch (const torch::Error &e) {                         \
    auto msg = e.what_without_backtrace();                \
    std::cout << "Torch Exception msg = " << msg << "\n"; \
    errVar.message = strdup(msg);                         \
    return retVal;                                        \
  }                                                       \
  catch (const std::exception &e) {                       \
    auto msg = e.what();                                  \
    std::cout << "Std Exception msg = " << msg << "\n";   \
    errVar.message = strdup(msg);                         \
    return retVal;                                        \
  }