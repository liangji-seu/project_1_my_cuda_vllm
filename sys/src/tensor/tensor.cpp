#include "tensor/tensor.h"
#include <cstring>//memcpy, memset这些操作
#include <glog/logging.h>
#include <algorithm>//std::remove_if
#include <numeric>//std::accumulate

namespace tensor{

    //降维累乘
    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    Tensor::Tensor(DataType_t data_type, 
                    std::vector<size_t> dims,

                    bool need_alloc,
                    std::shared_ptr<base::DeviceController> controller,
                    void* ptr):
            dims(std::move(dims)), data_type(data_type){
     
        //计算元素个数
        this->size = reduce_dimension(this->dims.begin(), this->dims.end(), size_t(1));
        if(need_alloc && controller){
            //需要自己分配, 且已经指定控制器
            self_allocate(controller); 
        } else{
            //无需分配，直接指定
            init_buffer(controller, this->data_type, ptr);
        }
    }





    //转移函数
    void Tensor::to(std::string dst_device_type, cudaStream_t stream){
        CHECK_NE(this->buffer, nullptr);
        auto device_type = this->get_device_type();
        if(dst_device_type == "cpu"){
            if(device_type == base::DeviceType_t::CPU){
                //就在cpu，无需变动
                return;
            } else if (device_type == base::DeviceType_t::GPU){
                size_t byte_size = this->get_byte_size();
                //在CPU端创建好buffer
                auto cpu_controller = base::CPUDeviceControllerFactory::get_instance();
                auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, nullptr, base::DeviceType_t::Unknown, cpu_controller, false);
            
                //拷贝数据
                cpu_controller->mem_copy(buffer->get_ptr(), cpu_buffer->get_ptr(), byte_size,
                                        base::DeviceType_t::GPU, base::DeviceType_t::CPU
                                        );
                //重定向该tensor的buffer
                this->buffer = cpu_buffer;
            }

        } else if (dst_device_type == "cuda"){
            if(device_type == base::DeviceType_t::CPU){
                //cpu->cuda
                size_t byte_size = this->get_byte_size();
                auto cu_alloc = base::GPUDeviceControllerFactory::get_instance();
                auto cu_buffer = std::make_shared<base::Buffer>(byte_size, nullptr, base::DeviceType_t::Unknown, cu_alloc, false);

                cu_alloc->mem_copy(buffer->get_ptr(), cu_buffer->get_ptr(), byte_size,
                                base::DeviceType_t::CPU, base::DeviceType_t::GPU,stream);

                this->buffer = cu_buffer;
            } else {
                //就在GPU，不需要变动。
                return;
            }

        } else {
            return;
        }
    }

    void Tensor::reshape(const std::vector<size_t>& dst_dims){

        //先换成一维维度
        size_t dst_size = reduce_dimension(dst_dims.begin(), dst_dims.end(), 1);
        if (!this->buffer) {
            //如果buffer是空的，直接赋值就行
            this->dims = dst_dims;
            this->size = dst_size;
            return;
        }


        if (dst_size > this->size) {
            //如果目标元素数量多了，要新开辟内存,然后替换上去
            auto new_buffer = std::make_shared<base::Buffer>(dst_size * DataTypeSize(this->data_type),
                                                            nullptr, base::DeviceType_t::Unknown,
                                                            buffer->get_controller(), false);
            new_buffer->buffer_self_allocate();
            new_buffer->buffer_copy_from(*(buffer.get()));
            this->buffer = new_buffer;
        }
        this->dims = dst_dims;
        this->size = dst_size;
    }


    void Tensor::reset(const std::vector<size_t>& dims,
                DataType_t data_type){
        this->data_type = data_type;
        this->dims = dims;
        this->size = reduce_dimension(dims.begin(), dims.end(), 1);
        this->buffer = nullptr;
    }

    bool Tensor::assign(std::shared_ptr<base::Buffer> dst_buffer){
        CHECK(dst_buffer != nullptr);
        if (this->buffer) {
            if (buffer->get_device_type() != dst_buffer->get_device_type()) {
            LOG(ERROR) << "The device type of the new buffer is different from the original one.";
            }
        }

        
        size_t byte_size = this->get_byte_size();
        if (byte_size > dst_buffer->get_byte_size()) {
            LOG(ERROR) << "The size of buffer is too small for the tensor!";
            return false;
        }
        this->buffer = dst_buffer;
        return true;
    }

    Tensor Tensor::clone() const{
        Tensor new_tensor = *this;
        size_t byte_size = this->get_byte_size();

        auto allocator = this->buffer->get_controller();
        new_tensor.buffer = std::make_shared<base::Buffer>(byte_size, nullptr, base::DeviceType_t::Unknown, allocator, false);
        new_tensor.buffer->buffer_copy_from(*this->buffer);
        return new_tensor;
    }


    bool Tensor::self_allocate(std::shared_ptr<base::DeviceController> allocator, bool need_realloc){
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null "
                        "pointer!";
            return false;
        }

        //获取该张量的字节大小
        size_t byte_size = this->get_byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
            return false;
        }

        //如果有指定的内存 且， 字节大小 小于当前指定的buffer的实际字节大小
        if (this->buffer && byte_size <= this->buffer->get_byte_size()) {
            if (!need_realloc) {
            return true;
            }
        }


        //直接创建一个新的buffer对象，构造函数里面，ptr == nullptr, 表明创建新的buffer对象。
        this->buffer = std::make_shared<base::Buffer>(byte_size, nullptr, base::DeviceType_t::Unknown, allocator, false);
        if (!this->buffer->get_ptr()) {
            LOG(ERROR) << "The memory allocated is a null pointer!";
            return false;
        }
        return true;
    }

    void Tensor::init_buffer(std::shared_ptr<base::DeviceController> allocator, DataType_t data_type, void* ptr){

        //直接利用现成的buffer对象来绑定，make_shared = new, 构造函数里面指定了ptr，说明存在现成的buffer对象，buffer的构造函数中直接赋值。
        if (!allocator) {
            std::shared_ptr<base::Buffer> buffer =
                std::make_shared<base::Buffer>(DataTypeSize(data_type) * this->size, ptr, base::DeviceType_t::Unknown, allocator, true);
            this->buffer = buffer;
        } else {
            self_allocate(allocator, true);
        }

    }


    

/**
 * 查询方法实现
 */
    bool Tensor::is_empty() const{
        return this->size ==0 ||
               this->dims.size() == 0 ||
               this->data_type == DataType_t::Unknown ||
               this->buffer == nullptr ||
               this->buffer->get_ptr() == nullptr;
    }

    std::shared_ptr<base::Buffer> Tensor::get_buffer() const{
        return buffer;
    }

    //张量的元素个数
    size_t Tensor::get_size() const { 
        return this->size; 
    }



    size_t Tensor::get_byte_size() const {
        return this->size * DataTypeSize(data_type);
    }

    size_t Tensor::get_dims_size() const {
        return static_cast<size_t>(dims.size());
    }

    DataType_t Tensor::get_data_type() const{
        return data_type;
    }

    size_t Tensor::get_dim(size_t axis) const{
        CHECK(axis >=0);
        CHECK(axis < this->dims.size());
        return this->dims.at(axis);
    }

    const std::vector<size_t>& Tensor::get_dims() const{
        return dims;
    }



    std::vector<size_t> Tensor::get_strides() const{
        std::vector<size_t> strides;
        if(!dims.empty()){
            for(size_t i = 0; i< dims.size() -1; ++i){
                size_t stride = reduce_dimension(dims.begin() + i + 1, dims.end(), 1);
                strides.push_back(stride);
            }

            strides.push_back(1);
        }
        return strides;
    }

    base::DeviceType_t Tensor::get_device_type() const{
        return this->buffer->get_device_type();
    }

    void* Tensor::get_ptr() {
        CHECK(this->buffer != nullptr);
        return this->buffer->get_ptr();
    }

    void* Tensor::get_ptr_offset(size_t offset) {
        CHECK(this->buffer != nullptr);
        size_t type_size = DataTypeSize(data_type);
        return static_cast<char*>(this->buffer->get_ptr()) + offset * type_size;
    }


    //补充其他张量层的方法：
    void Tensor::unsqueeze(size_t axis){
        CHECK(axis >= 0);
        CHECK(axis < this->dims.size());
        this->dims.insert(this->dims.begin() + axis, 1);
    }

    void Tensor::squeeze(size_t axis){
        if(axis == -1){
            // 删除所有大小为1的维度
            this->dims.erase(
                std::remove_if(this->dims.begin(), this->dims.end(),
                               [](size_t d) { return d == 1; }),
                this->dims.end());
        } else {
            CHECK(axis >= 0 && axis < this->dims.size());
            CHECK_EQ(this->dims[axis], 1) << "The dimension to squeeze must be 1";
            this->dims.erase(this->dims.begin() + axis);
        }
        // 元素总数不变，size 不需要更新
    }


    Tensor Tensor::slice(size_t axis, size_t start, size_t end, size_t step){
        CHECK(axis < this->dims.size());
        CHECK(step > 0);
        auto num = this->dims[axis];
        CHECK(start < num && end <= num && start < end);

        // 切片轴的新维度
        size_t new_dim = (end - start + step - 1) / step;

        // 计算 inner_size: axis 之后所有维度的乘积（字节数）
        auto dims_after = std::vector<size_t>(this->dims.begin() + axis + 1, this->dims.end());
        size_t inner_elements = dims_after.empty() ? 1 : reduce_dimension(dims_after.begin(), dims_after.end(), size_t(1));
        size_t element_size = DataTypeSize(data_type);
        size_t inner_bytes = inner_elements * element_size;

        // 计算 outer_count: axis 之前所有维度的乘积
        size_t outer_count = 1;
        for (size_t i = 0; i < axis; ++i) {
            outer_count *= this->dims[i];
        }

        // 旧 axis 跨度（字节）：num * inner_bytes
        size_t old_axis_byte_stride = num * inner_bytes;

        // 新 axis 跨度（字节）：new_dim * inner_bytes
        size_t new_axis_byte_stride = new_dim * inner_bytes;

        // 构建新 dims
        auto new_dims = this->dims;
        new_dims[axis] = new_dim;

        // 分配新 buffer
        size_t new_byte_size = outer_count * new_axis_byte_stride;
        auto allocator = this->buffer->get_controller();
        auto new_buffer = std::make_shared<base::Buffer>(
            new_byte_size, nullptr, base::DeviceType_t::Unknown, allocator, false);

        // 逐行拷贝
        auto* src_base = static_cast<unsigned char*>(this->buffer->get_ptr());
        auto* dst_base = static_cast<unsigned char*>(new_buffer->get_ptr());

        for (size_t outer = 0; outer < outer_count; outer++) {
            size_t slice_idx = 0;
            for (size_t s = start; s < end; s += step) {
                auto* src = src_base + outer * old_axis_byte_stride + s * inner_bytes;
                auto* dst = dst_base + outer * new_axis_byte_stride + slice_idx * inner_bytes;
                std::memcpy(dst, src, inner_bytes);
                slice_idx++;
            }
        }

        // 构建返回的 Tensor
        Tensor result(this->data_type, new_dims, false, allocator, nullptr);
        result.buffer = new_buffer;
        result.size = outer_count * new_dim * inner_elements;
        return result;
    }


















}