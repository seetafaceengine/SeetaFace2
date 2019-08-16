#ifndef _SEETANET_COMMON_H_
#define _SEETANET_COMMON_H_


struct SeetaNetDataSize
{
    std::vector<int> data_dim;
    SeetaNetDataSize() {
        data_dim.clear();
    };

    SeetaNetDataSize( const SeetaNetDataSize &a ) {
        data_dim = a.data_dim;
    };
};

#endif