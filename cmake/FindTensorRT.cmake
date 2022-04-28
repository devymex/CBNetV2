# FindTensorRT
# ------------
#	You can specify the path to TensorRT root in TensorRT_ROOT_DIR
#
#	This will define the following variables:
#	TensorRT_FOUND			- True if the found the TensorRT library
#	TensorRT_INC_DIR		- TensorRT include directory
#	TensorRT_LIB_DIR		- TensorRT library directory
#	TensorRT_LIBRARIES		- TensorRT libraries
#	TensorRT_VERSION		- TensorRT Version (x.x.x)

IF(TensorRT_ROOT_DIR)
    FIND_PATH(TensorRT_INC_DIR "NvInfer.h"
        HINTS "${TensorRT_ROOT_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}"
        PATH_SUFFIXES "include"
        )
    FIND_PATH(TensorRT_LIB_DIR "libnvinfer.so"
        HINTS "${TensorRT_ROOT_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}"
        PATH_SUFFIXES "lib" "lib64"
        )
    IF(EXISTS "${TensorRT_INC_DIR}" AND EXISTS "${TensorRT_LIB_DIR}")
        IF(${TRT_USE_STATIC_LIB})
            SET(SUFFIX "_static")
        ENDIF()

        FIND_LIBRARY(nvinfer_LIB_FILE			NAMES "nvinfer${SUFFIX}"
            HINTS "${TensorRT_LIB_DIR}")
        FIND_LIBRARY(nvinfer_plugin_LIB_FILE	NAMES "nvinfer_plugin${SUFFIX}"
            HINTS "${TensorRT_LIB_DIR}")
        FIND_LIBRARY(nvparsers_LIB_FILE			NAMES "nvparsers${SUFFIX}"
            HINTS "${TensorRT_LIB_DIR}")
        FIND_LIBRARY(nvonnxparser_LIB_FILE		NAMES "nvonnxparser${SUFFIX}"
            HINTS "${TensorRT_LIB_DIR}")
        FIND_LIBRARY(protobuf_LIB_FILE			NAMES "protobuf${SUFFIX}"
            HINTS "${TensorRT_LIB_DIR}")

        SET(TensorRT_LIBRARIES ${nvinfer_LIB_FILE}
            ${nvinfer_LIB_FILE}
            ${nvinfer_plugin_LIB_FILE}
            ${nvparsers_LIB_FILE}
            ${nvonnxparser_LIB_FILE}
            ${protobuf_LIB_FILE}
            )

        FILE(READ ${TensorRT_INC_DIR}/NvInferVersion.h NVINFER_HDR_CONTENTS)

        STRING(REGEX MATCH "define NV_TENSORRT_MAJOR * +([0-9]+)"
            TensorRT_MAJOR_VERSION "${NVINFER_HDR_CONTENTS}")
        STRING(REGEX REPLACE "define NV_TENSORRT_MAJOR * +([0-9]+)" "\\1"
            TensorRT_MAJOR_VERSION "${TensorRT_MAJOR_VERSION}")

        STRING(REGEX MATCH "define NV_TENSORRT_MINOR * +([0-9]+)"
            TensorRT_MINOR_VERSION "${NVINFER_HDR_CONTENTS}")
        STRING(REGEX REPLACE "define NV_TENSORRT_MINOR * +([0-9]+)" "\\1"
            TensorRT_MINOR_VERSION "${TensorRT_MINOR_VERSION}")

        STRING(REGEX MATCH "define NV_TENSORRT_PATCH * +([0-9]+)"
            TensorRT_PATCH_VERSION "${NVINFER_HDR_CONTENTS}")
        STRING(REGEX REPLACE "define NV_TENSORRT_PATCH * +([0-9]+)" "\\1"
            TensorRT_PATCH_VERSION "${TensorRT_PATCH_VERSION}")

        SET(TensorRT_VERSION ${TensorRT_MAJOR_VERSION}.${TensorRT_MINOR_VERSION}.${TensorRT_PATCH_VERSION})
    ENDIF()
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TensorRT
    FOUND_VAR		TensorRT_FOUND
    REQUIRED_VARS	TensorRT_INC_DIR
    REQUIRED_VARS	TensorRT_LIBRARIES
    VERSION_VAR		TensorRT_VERSION
    )