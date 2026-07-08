#pragma once

#include "layer.h"
#include "base/base.h"

namespace op{


    class EmbeddingLayer : public LayerParam{
        public:
            explicit EmbeddingLayer(
                base::DeviceType_t device_type = base::DeviceType_t::Unknown
            );

            //自检
            base::error::Status check_layer() override;

            //重定向该算子的后端接口获取
            base::error::Status forward() override;
    };


}