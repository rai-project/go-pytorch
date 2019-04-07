#pragma once

#define HANDLE_TH_ERRORS try {
#define END_HANDLE_TH_ERRORS(errVar, retVal)  \
  }                                           \
  catch (const torch::Error &e) {             \
    auto msg = e.what_without_backtrace();    \
    auto err = Torch_Error{                   \
        .message = new char[strlen(msg) + 1], \
    };                                        \
    std::strcpy(err.message, msg);            \
    *errVar = err;                            \
    return retVal;                            \
  }                                           \
  catch (const std::exception &e) {           \
    auto msg = e.what();                      \
    auto err = Torch_Error{                   \
        .message = new char[strlen(msg) + 1], \
    };                                        \
    std::strcpy(err.message, msg);            \
    *errVar = err;                            \
    return retVal;                            \
  }
