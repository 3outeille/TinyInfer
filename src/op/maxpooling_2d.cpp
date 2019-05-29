#include "op/maxpooling_2d.hpp"
#include "runtime/kernel/maxpool.hpp"

namespace tinyinfer{
    namespace op{
        Maxpooling2dOp::Maxpooling2dOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("Maxpooling2dOp", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void Maxpooling2dOp::set_kernel_x(int x) {
            m_kernel_x = x;
        }

        void Maxpooling2dOp::set_kernel_y(int y) {
            m_kernel_y = y;
        }

        void Maxpooling2dOp::set_stride_x(int x) {
            m_stride_x = x;
        }

        void Maxpooling2dOp::set_stride_y(int y) {
            m_stride_y = y;
        }

        void Maxpooling2dOp::forward() {
            this->get_outputs().at(0).set_tensor_ptr(
                    runtime::kernel::Maxpool(this->get_inputs().at(0).get_tensor_ptr(),
                                             this->m_kernel_x, this->m_kernel_y,
                                             this->m_stride_x, this->m_stride_y));
        }
    }
}