#ifndef _SEETANET_LAYER_TYPE_H_
#define _SEETANET_LAYER_TYPE_H_

namespace seeta
{
	enum Enum_SeetaNetLayerType
	{
		Enum_ConvolutionLayer = 0,
		Enum_EltwiseLayer = 1,
		Enum_ConcatLayer = 2,
		Enum_ExpLayer = 3,
		Enum_InnerProductLayer = 4,
		Enum_LRNLayer = 5,
		Enum_MemoryDataLayer = 6,
		Enum_PoolingLayer = 7,
		Enum_PowerLayer = 8,
		Enum_ReLULayer = 9,
		Enum_SoftmaxLayer = 10,
		Enum_SliceLayer = 11,
		Enum_BatchNormliseLayer = 13,
		Enum_ScaleLayer = 14,
		Enum_SplitLayer = 15,
		Enum_PreReLULayer = 16,
		Enum_DeconvolutionLayer = 17,
		Enum_CropLayer = 18,
		Enum_SigmoidLayer = 19,

		// tf convert operator
		Enum_SpaceToBatchNDLayer = 20,
		Enum_BatchToSpaceNDLayer = 21,

		// tf reshape
		Enum_ReshapeLayer = 22,
		Enum_RealMulLayer = 23,
		Enum_ShapeIndexPatchLayer = 31,
		Enum_FinallyLayer = 1001,
	};
};

#endif
