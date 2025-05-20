#include <glog/logging.h>
#include <armadillo>

#include "add_kernel.h"

namespace kernel {

void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, void* stream) {
    // 判空
    CHECK_EQ(input1.empty(), false);
    CHECK_EQ(input2.empty(), false);
    CHECK_EQ(output.empty(), false);

    // 检查大小
    CHECK_EQ(input1.size(), input2.size());
    CHECK_EQ(input1.size(), output.size());

    // 不复制，向量将在其生命周期内绑定到辅助存储器（ptr）; 无法更改向量中的元素数
    arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
    
    // 计算
    output_vec = input_vec1 + input_vec2;
}

} // namespace kernel