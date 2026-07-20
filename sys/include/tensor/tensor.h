#pragma once
#include <glog/logging.h>
#include "base/base.h"
#include "base/Buffer.h"

#include <cuda_runtime.h>

#include <vector>
#include <memory> //智能指针
#include <armadillo> //线性代数
#include <string>

namespace tensor{
    enum class DataType_t : uint8_t {
        Unknown = 0,
        fp32 = 1,
        int8 = 2,
        int32 = 3,
    };

    inline size_t DataTypeSize(DataType_t data_type) {
        if (data_type == DataType_t::fp32) {
            return sizeof(float);
        } else if (data_type == DataType_t::int8) {
            return sizeof(int8_t);
        } else if (data_type == DataType_t::int32) {
            return sizeof(int32_t);
        } else {
            return 0;
        }
    }

    class Tensor {
        private:
            std::vector<size_t> dims; //张量shape
            size_t size;              //元素个数
            DataType_t data_type = DataType_t::Unknown;//每个元素数据类型

            std::shared_ptr<base::Buffer> buffer;//底层内存 RAII 
        public:
            explicit Tensor() = default;
            ~Tensor() = default; //无需再次释放资源，资源已经交给buffer的析构+shared_ptr来管理了
            explicit Tensor(DataType_t data_type, 
                            std::vector<size_t> dims,

                            bool need_alloc = false,
                            std::shared_ptr<base::DeviceController> controller = nullptr,
                            void* ptr = nullptr);

            /**
             * 查询方法
             */
            bool is_empty() const;
            std::shared_ptr<base::Buffer> get_buffer() const;
            size_t get_size() const;//张量元素个数
            size_t get_byte_size() const;//张量总的字节大小
            size_t get_dims_size() const;//张量维度，有多少个轴
            DataType_t get_data_type() const;//元素数据类型
            size_t get_dim(size_t axis) const;//axis这一个轴的维数/数量
            const std::vector<size_t>& get_dims() const;//获取张量维度
            std::vector<size_t> get_strides() const; //获取+1时一维索引中的跨度
            base::DeviceType_t get_device_type() const;
            void set_device_type(base::DeviceType_t device_type);

            void* get_ptr(); //获取底层的内存指针
            const void* get_ptr() const; //const版本
            void* get_ptr_offset(size_t offset); 

            //一维索引访问张量内存元素
            template<typename T>
            T& peek_index(size_t offset);

            //坐标访问张量内存元素
            template<typename T>
            T& peek_position(std::vector<size_t> pos);


            /**
             * 操作方法
             */
            void to(std::string dst_device_type, cudaStream_t stream);//转移张量设备
            void reshape(const std::vector<size_t>& dims);//改成指定形状
            void reset(const std::vector<size_t>& dims,//重置成空有形状，无内存
                        DataType_t data_type = DataType_t::int32);
            bool assign(std::shared_ptr<base::Buffer> buffer);//替换张量底层数据
            Tensor clone() const; //深拷贝

        private:
            bool self_allocate(std::shared_ptr<base::DeviceController> allocator,bool need_realloc = false);//需要自行分配内存，执行分配动作
            void init_buffer(std::shared_ptr<base::DeviceController> allocator, DataType_t data_type,void* ptr);    //直接指定现成的内存
                
        public:
            //补充其他张量层的方法：
            void unsqueeze(size_t axis);
            void squeeze(size_t axis = -1);//移除所有轴（-1），指定轴，大小为1的维度
            Tensor slice(size_t axis, size_t start, size_t end, size_t step);


    };

    template<typename T>
    T& Tensor::peek_index(size_t offset){
        auto start = static_cast<T*>(this->buffer->get_ptr()) + offset;
        return *start;
    }

    
    //坐标访问张量内存元素
    template<typename T>
    T& Tensor::peek_position(std::vector<size_t> pos){
        size_t offset = 0;

        const auto& strides = get_strides();
        for(size_t i = 0; i< pos.size(); i++){
            offset +=pos[i]*strides[i];
        }
        return peek_index<T>(offset);
    }




}