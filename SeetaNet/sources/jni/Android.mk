LOCAL_PATH := $(call my-dir)


include $(CLEAR_VARS)

LOCAL_MODULE := seetanet2

MY_CPP_LIST := $(wildcard $(LOCAL_PATH)/../src/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/mem/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/proto/*.cc)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/mem/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/sync/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/orz/tools/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/.. \
		    $(LOCAL_PATH)/../include \
                    $(LOCAL_PATH)/../src/include_inner \
                    $(LOCAL_PATH)/../src/include_inner/layers \
                    $(LOCAL_PATH)/../src/mem \
                    $(LOCAL_PATH)/../thirdParty/Android/include/ \
                    $(LOCAL_PATH)/../thirdParty/Android/protobuf/include \
                    $(LOCAL_PATH)/../thirdParty/Android/openblas/include \
                    $(LOCAL_PATH)/../src/proto \
		            $(LOCAL_PATH)/../src

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib -fuse-ld=bfd

LOCAL_CFLAGS += -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize -ffast-math

LOCAL_LDLIBS += -lc -llog -latomic -lm
include $(BUILD_SHARED_LIBRARY)
