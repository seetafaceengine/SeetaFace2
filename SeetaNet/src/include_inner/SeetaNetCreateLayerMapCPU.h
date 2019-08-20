#ifndef _SEETANET_CREATE_LAYER_MAP_CPU_H_
#define _SEETANET_CREATE_LAYER_MAP_CPU_H_


#include <map>

#include "SeetaNetLayerType.h"
#include "SeetaNetProto.h"
#include "SeetaNetCreateLayerDetailCPU.h"

#include <stdint.h>

using namespace seeta;

template<typename DType>
class SeetaNetBaseLayer;

template<typename DType>
class CreateLayerMapCPU {
public:
    typedef int CREATE_NET_PARSEFUNCTION( SeetaNetBaseLayer<DType> *&, SeetaNet_LayerParameter &, SeetaNetResource<DType> * );

    static int( *FindRunFunciton( int32_t layertype ) )( SeetaNetBaseLayer<DType> *&, SeetaNet_LayerParameter &, SeetaNetResource<DType> * ) {

        auto iteraiton_current = m_parse_function_map.find( layertype );

        int( *pfun )( SeetaNetBaseLayer<DType> *&, SeetaNet_LayerParameter &, SeetaNetResource<DType> * )( nullptr );

        if( iteraiton_current != m_parse_function_map.end() ) {
            pfun = iteraiton_current->second;

        }

        return pfun;
    }
private:
    static std::map<int32_t, int( * )( SeetaNetBaseLayer<DType>* &, SeetaNet_LayerParameter &, SeetaNetResource<DType>* )> CreateFunctionMap() {

        std::map<int32_t, int( * )( SeetaNetBaseLayer<DType>* &, SeetaNet_LayerParameter &, SeetaNetResource<DType>* )> FunctionMap;
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_MemoryDataLayer, CreateMemoryDataFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ConvolutionLayer, CreateConvolutionFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ReLULayer, CreateReluFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_PoolingLayer, CreatePoolingFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_InnerProductLayer, CreateInnerproductionFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_SoftmaxLayer, CreateSoftmaxFunctionCPU<DType> ) );

        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_EltwiseLayer, CreateEltwiseFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ConcatLayer, CreateConcatFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ExpLayer, CreateExpFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_PowerLayer, CreatePowerFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_BatchNormliseLayer, CreateBatchNormliseFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ScaleLayer, CreateScaleFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_SplitLayer, CreateSplitFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_PreReLULayer, CreatePreReLUFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_DeconvolutionLayer, CreateDeconvolutionFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_CropLayer, CreateCropLayerFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_SigmoidLayer, CreateSigmoidFunctionCPU<DType> ) );

        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_SpaceToBatchNDLayer, CreateSpaceToBatchNDFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_BatchToSpaceNDLayer, CreateBatchToSpaceNDFunctionCPU<DType> ) );

        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ReshapeLayer, CreateReshapeFunctionCPU<DType> ) );
        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_RealMulLayer, CreateRealMulFunctionCPU<DType> ) );

        FunctionMap.insert( std::pair<int32_t, CREATE_NET_PARSEFUNCTION *>( seeta::Enum_ShapeIndexPatchLayer, CreateShapeIndexPatchFunctionCPU<DType> ) );
        return FunctionMap;
    };
    static const std::map<int32_t, int( * )( SeetaNetBaseLayer<DType>* &, SeetaNet_LayerParameter &, SeetaNetResource<DType> * )> m_parse_function_map;
};

template<> const std::map<int32_t, int( * )( SeetaNetBaseLayer<float>* &, SeetaNet_LayerParameter &, SeetaNetResource<float>* )> CreateLayerMapCPU<float>::m_parse_function_map = CreateLayerMapCPU<float>::CreateFunctionMap();
template<> const std::map<int32_t, int( * )( SeetaNetBaseLayer<double>* &, SeetaNet_LayerParameter &, SeetaNetResource<double>* )> CreateLayerMapCPU<double>::m_parse_function_map = CreateLayerMapCPU<double>::CreateFunctionMap();


#endif
