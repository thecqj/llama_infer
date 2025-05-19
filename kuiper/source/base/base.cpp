#include "base/base.h"

namespace base {
namespace error {

Status Success(const std::string& err_msg) {
    return {kSuccess, err_msg};
}

Status FunctionNotImplement(const std::string& err_msg) {
    return {kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
    return {kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
    return {kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
    return {kInternalError, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
    return {kKeyValueHasExist, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
    return {kInvalidArgument, err_msg};
}

}   // namespace error
} // namespace base