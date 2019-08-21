LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := seetanet-prebuilt
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../SeetaNet/libs/$(TARGET_ARCH_ABI)/libseetanet2.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../SeetaNet/include/
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := SeetaFaceRecognizer2

MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../seeta/*.cpp)
MY_CPP_LIST += $(wildcard $(LOCAL_PATH)/../src/seeta/*.cpp)

LOCAL_SRC_FILES := $(MY_CPP_LIST:$(LOCAL_PATH)/%=%)

LOCAL_C_INCLUDES += $(LOCAL_PATH)/../
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../seeta/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/
LOCAL_C_INCLUDES += $(LOCAL_PATH)/../include/seeta/

LOCAL_LDFLAGS += -L$(LOCAL_PATH)/lib -fuse-ld=bfd

LOCAL_LDLIBS += -llog -lz

LOCAL_CFLAGS += -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize  -ffast-math

LOCAL_SHARED_LIBRARIES += seetanet-prebuilt

include $(BUILD_SHARED_LIBRARY)
