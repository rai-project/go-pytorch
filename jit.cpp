#ifdef ENABLE_PYTROCH_JIT

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

extern Torch_IValue Torch_ConvertIValueToTorchIValue(torch::IValue value);

struct Torch_JITModule {
  std::shared_ptr<torch::jit::script::Module> module;
};

struct Torch_JITModule_Method {
  torch::jit::script::Method& run;
};

Torch_JITModuleContext Torch_CompileTorchScript(char* cstring_script, Torch_Error* error) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  std::string script(cstring_script);
  auto mod = new Torch_JITModule();
  mod->module = torch::jit::compile(script);

  return (void*)mod;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, NULL)
}

Torch_JITModuleContext Torch_LoadJITModule(char* cstring_path, Torch_Error* error) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  std::string module_path(cstring_path);
  auto mod = new Torch_JITModule();
  mod->module = torch::jit::load(module_path);

  return (void*)mod;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, NULL)
}

void Torch_ExportJITModule(Torch_JITModuleContext ctx, char* cstring_path, Torch_Error* error) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  std::string module_path(cstring_path);
  auto mod = (Torch_JITModule*)ctx;
  mod->module->save(module_path);
  END_HANDLE_TH_ERRORS(Torch_GlobalError, )
}

Torch_JITModuleMethodContext Torch_JITModuleGetMethod(Torch_JITModuleContext ctx, char* cstring_method,
                                                      Torch_Error* error) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  std::string method_name(cstring_method);
  auto mod = (Torch_JITModule*)ctx;

  auto met = new Torch_JITModule_Method{mod->module->get_method(method_name)};

  return (void*)met;
  END_HANDLE_TH_ERRORS(Torch_GlobalError, NULL)
}

char** Torch_JITModuleGetMethodNames(Torch_JITModuleContext ctx, size_t* len) {
  auto mod = (Torch_JITModule*)ctx;
  auto size = mod->module->get_methods().size();
  *len = size;
  auto result = (char**)malloc(sizeof(char*) * size);

  int i = 0;
  for (auto& method : mod->module->get_methods()) {
    auto key = method.value()->name();
    auto ckey = new char[key.length() + 1];
    strcpy(ckey, key.c_str());

    *(result + i) = ckey;

    i++;
  }

  return result;
}

Torch_IValue Torch_JITModuleMethodRun(Torch_JITModuleMethodContext ctx, Torch_IValue* inputs, size_t input_size,
                                      Torch_Error* error) {
  HANDLE_TH_ERRORS(Torch_GlobalError);
  auto met = (Torch_JITModule_Method*)ctx;

  std::vector<torch::IValue> inputs_vec;

  for (int i = 0; i < input_size; i++) {
    auto ival = *(inputs + i);
    inputs_vec.push_back(Torch_ConvertTorchIValueToIValue(ival));
  }

  auto res = met->run(inputs_vec);
  return Torch_ConvertIValueToTorchIValue(res);
  END_HANDLE_TH_ERRORS(Torch_GlobalError, Torch_IValue{})
}

Torch_ModuleMethodArgument* Torch_JITModuleMethodArguments(Torch_JITModuleMethodContext ctx, size_t* res_size) {
  auto met = (Torch_JITModule_Method*)ctx;
  auto schema = met->run.getSchema();
  auto arguments = schema.arguments();

  auto result = (Torch_ModuleMethodArgument*)malloc(sizeof(Torch_ModuleMethodArgument) * arguments.size());
  *res_size = arguments.size();

  for (std::vector<torch::Argument>::size_type i = 0; i != arguments.size(); i++) {
    auto name = arguments[i].name();
    char* cstr_name = new char[name.length() + 1];
    strcpy(cstr_name, name.c_str());

    auto type = arguments[i].type()->str();
    char* cstr_type = new char[type.length() + 1];
    strcpy(cstr_type, type.c_str());

    *(result + i) = Torch_ModuleMethodArgument{
        .name = cstr_name,
        .typ = cstr_type,
    };
  }

  return result;
}

Torch_ModuleMethodArgument* Torch_JITModuleMethodReturns(Torch_JITModuleMethodContext ctx, size_t* res_size) {
  auto met = (Torch_JITModule_Method*)ctx;
  auto schema = met->run.getSchema();
  auto arguments = schema.returns();

  auto result = (Torch_ModuleMethodArgument*)malloc(sizeof(Torch_ModuleMethodArgument) * arguments.size());
  *res_size = arguments.size();

  for (std::vector<torch::Argument>::size_type i = 0; i != arguments.size(); i++) {
    auto name = arguments[i].name();
    char* cstr_name = new char[name.length() + 1];
    strcpy(cstr_name, name.c_str());

    auto type = arguments[i].type()->str();
    char* cstr_type = new char[type.length() + 1];
    strcpy(cstr_type, type.c_str());

    *(result + i) = Torch_ModuleMethodArgument{
        .name = cstr_name,
        .typ = cstr_type,
    };
  }

  return result;
}

void Torch_DeleteJITModuleMethod(Torch_JITModuleMethodContext ctx) {
  auto med = (Torch_JITModule_Method*)ctx;
  delete med;
}

void Torch_DeleteJITModule(Torch_JITModuleContext ctx) {
  auto mod = (Torch_JITModule*)ctx;
  delete mod;
}
#endif  // ENABLE_PYTROCH_JIT