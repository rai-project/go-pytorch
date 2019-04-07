#pragma once

#define HANDLE_TH_ERRORS(errVar)     \
  try {                              \
    if (errVar.message != nullptr) { \
      free(errVar.message);          \
    }                                \
    errVar.message = 0;
#define END_HANDLE_TH_ERRORS(errVar, retVal) \
  }                                          \
  catch (const torch::Error &e) {            \
    auto msg = e.what_without_backtrace();   \
    std::strcpy(errVar.message, msg);        \
    return retVal;                           \
  }                                          \
  catch (const std::exception &e) {          \
    auto msg = e.what();                     \
    std::strcpy(errVar.message, msg);        \
    return retVal;                           \
  }
