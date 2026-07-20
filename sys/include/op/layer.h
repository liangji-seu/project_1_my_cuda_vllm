#pragma once

#include <string>

#include "base/base.h"
#include "tensor/tensor.h"
#include "base/cuda_stream.h"


//算子层
namespace op{

    //不同层的类型的注册
    enum class LayerType_t : uint8_t {
        Unknown = 0,
        Linear = 1,
        Encode = 2,
        Embedding = 3,
        RMSNorm = 4,
        Matmul = 5,
        RoPe = 6,
        MHA = 7,
        Softmax = 8,
        Add = 9,
        SwiGLU = 10,
    };
    
    /**
     * 层的基类
     * 一个层，要有：
     * - 输入张量
     * - 输出张量
     * - 权重张量
     * - 运算过程
     */
    class BaseLayer{
        //名字 + 层类型 + 数据类型 + 设备类型
        protected:
            std::string name;//层的名字
            LayerType_t layer_type = LayerType_t::Unknown;//xxx层
            tensor::DataType_t data_type = tensor::DataType_t::Unknown;//这个层处理的数据类型
            base::DeviceType_t device_type = base::DeviceType_t::Unknown;//这个层工作的设备

        public:
            explicit BaseLayer(
                base::DeviceType_t device_type,
                LayerType_t layer_type,
                tensor::DataType_t data_type,
                std::string layer_name = "default-BaseLayer"
            );

            /**
             * 查询方法
             */
            tensor::DataType_t get_data_type() const;//查询层处理的数据类型
            LayerType_t get_layer_type() const;//查询层类型
            const std::string& get_layer_name() const; //层的名字
            base::DeviceType_t get_device_type() const; //层的工作设备

            virtual size_t get_input_tensor_num() const = 0;//查询有几个输入张量
            virtual size_t get_output_tensor_num() const = 0;//查询有几个输出张量


            /**
             * 操作方法
             */
            virtual base::error::Status init() = 0;//层的初始化
            //前向传播
            virtual base::error::Status forward() = 0;
            virtual base::error::Status forward(const tensor::Tensor& input1,
                                        const tensor::Tensor& output1) = 0;

            virtual base::error::Status forward(const tensor::Tensor& input1,
                                        const tensor::Tensor& input2,
                                        const tensor::Tensor& output1) = 0;

            virtual base::error::Status forward(const tensor::Tensor& input1,
                                        const tensor::Tensor& input2,
                                        const tensor::Tensor& input3,
                                        const tensor::Tensor& output1) = 0;

            virtual base::error::Status forward(const tensor::Tensor& input1,
                                        const tensor::Tensor& input2,
                                        const tensor::Tensor& input3,
                                        const tensor::Tensor& input4,
                                        const tensor::Tensor& output1) = 0;

            virtual base::error::Status forward(const tensor::Tensor& input1,
                                        const tensor::Tensor& input2,
                                        const tensor::Tensor& input3,
                                        const tensor::Tensor& input4,
                                        const tensor::Tensor& input5, 
                                        const tensor::Tensor& output1) = 0;

            //设置张量
            virtual void set_input_tensor(size_t index, const tensor::Tensor& input) = 0;
            virtual void set_output_tensor(int32_t index, const tensor::Tensor& output) = 0;
            virtual base::error::Status set_weight_tensor(size_t index, const tensor::Tensor& weight) = 0;


            //检查张量
            virtual base::error::Status check_layer() = 0;//检查输入张量们，输出张量们
            

            //获取张量
            virtual tensor::Tensor& get_input(size_t index) = 0;
            virtual tensor::Tensor& get_output(size_t index) = 0;

            //设置一些属性
            void set_layer_name(const std::string& layer_name);
            void set_device_type(base::DeviceType_t device_type);
    };



    /**
     * 无权重的层
     */
    class Layer : public BaseLayer{
        //新增属性， 这个Layer才是一个真正的层
        //输入张量 + 输出张量 + cuda流
        protected:
            std::vector<tensor::Tensor> inputs;//输入张量组
            std::vector<tensor::Tensor> outputs;//输出张量组
            std::shared_ptr<kernel::CudaStream> cuda_stream;//层的cuda工作流

        public:
            explicit Layer(
                base::DeviceType_t device_type,
                LayerType_t layer_type,
                tensor::DataType_t data_type = tensor::DataType_t::fp32,
                std::string layer_name = "default-Layer"
            );



            /**
             * 操作函数
             */
            base::error::Status init() override;

            
            //自检
            base::error::Status check_layer() override;


            //前向传播
            base::error::Status forward() override;

            base::error::Status forward(const tensor::Tensor& input1,
                                const tensor::Tensor& output1) override;

            base::error::Status forward(const tensor::Tensor& input1,
                                const tensor::Tensor& input2,
                                const tensor::Tensor& output1) override;

            base::error::Status forward(const tensor::Tensor& input1,
                                const tensor::Tensor& input2,
                                const tensor::Tensor& input3,
                                const tensor::Tensor& output1) override;

            base::error::Status forward(const tensor::Tensor& input1,
                                const tensor::Tensor& input2,
                                const tensor::Tensor& input3,
                                const tensor::Tensor& input4,
                                const tensor::Tensor& output1) override;

            base::error::Status forward(const tensor::Tensor& input1,
                                const tensor::Tensor& input2,
                                const tensor::Tensor& input3,
                                const tensor::Tensor& input4,
                                const tensor::Tensor& input5,
                                const tensor::Tensor& output1) override;


            
            //设置张量(无权重)
            void set_input_tensor(size_t index, const tensor::Tensor& input) override;
            void set_output_tensor(int32_t index, const tensor::Tensor& output) override;
            base::error::Status set_weight_tensor(size_t index, const tensor::Tensor& weight) override;

            //检查张量
            base::error::Status check_tensor(const tensor::Tensor& tensor, bool is_input = true) const;

            //获取张量（无权重）
            tensor::Tensor& get_input(size_t index) override;
            tensor::Tensor& get_output(size_t index) override;
            //tensor::Tensor& get_weight_tensor

            //多输入多输出数量
            size_t get_input_tensor_num() const override;
            size_t get_output_tensor_num() const override;

            //重新设置几个输入口，输出口
            void reset_input_tensor_num(size_t size);
            void reset_output_tensor_num(size_t size);

            void set_cuda_stream(std::shared_ptr<kernel::CudaStream> stream);

            //获取该层的流
            std::shared_ptr<kernel::CudaStream> get_cuda_stream() const;

    };















    /**
     * 有权重的层
     */
    class LayerParam : public Layer {
        //除了有普通无权重的层：输入输出，流，另外多了一个
        //权重张量 + 量化张量
        protected:
            std::vector<tensor::Tensor> weights;//权重张量
            bool is_quant_layer = false;        //是否是量化层
            tensor::Tensor scales;              //量化缩放张量
            size_t group_size = 0;              //?

        public:
            explicit LayerParam(
                base::DeviceType_t device_type,
                LayerType_t layer_type,
                bool is_quant_layer = false,
                std::string layer_name = "default-LayerParam"
            );

            /**
             * 查询方法
             */
            size_t get_weight_tensor_num() const;//权重张量的个数
            size_t get_scale_num() const; //量化张量的个数？

            /**
             * 操作方法
             */
            void reset_weight_tensor_num(size_t size);//修改权重张量的数量
            tensor::Tensor& get_weight(size_t index);//获取某一个权重张量

            //转移
            void to(std::string dst_device_type);

            //设置权重张量
            base::error::Status set_weight_tensor(size_t index, const tensor::Tensor& weight) override;

            //从原始指针直接设置权重（模型加载时使用）
            base::error::Status set_weight(size_t idx, const std::vector<size_t>& dims,
                                           const void* ptr, base::DeviceType_t device_type);

            //设置量化张量
            void set_scales(const tensor::Tensor& scales);

            //设置组元素个数
            void set_group_size(size_t group_size);

    };




}