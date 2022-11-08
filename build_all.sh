SDK_PATH="/home/marc/rv1126_rv1109_linux_sdk_v1.8.0_20210224/"
echo "SDK_PATH: $SDK_PATH"
GCC_PATH=$SDK_PATH/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin
echo "GCC_PATH: $GCC_PATH"
SYS_ROOT=$SDK_PATH/buildroot/output/rockchip_rv1126_rv1109/host/arm-buildroot-linux-gnueabihf/sysroot
echo "SYS_ROOT: $SYS_ROOT"
echo "PWD: $PWD"
ROOT_LIB_PATH=${PWD}/libs/
echo "ROOT_LIB_PATH: $ROOT_LIB_PATH"
echo "Start build"
echo "Build lib first"

RK1126_TOOL_CHAIN=$SDK_PATH/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf
GCC_COMPILER_PREFIX=${RK1126_TOOL_CHAIN}/bin/arm-linux-gnueabihf
echo "GCC_COMPILER_PREFIX: $GCC_COMPILER_PREFIX"

# LIB_BUILD_DIR=${PWD}/yolo_detect/build
#echo "Build dir: ${LIB_BUILD_DIR}"

echo "clean old static lib file"
rm -rf ${PWD}/libs/*.a
echo "clean old static lib file done"

cd yolo_detect/build/
# cd ${BUILD_DIR}
cmake ../ -DCMAKE_C_COMPILER=${GCC_COMPILER_PREFIX}-gcc -DCMAKE_CXX_COMPILER=${GCC_COMPILER_PREFIX}-g++
make -j4
mv librknnyolo.a $ROOT_LIB_PATH

cd ..
cd ..
echo "PWD: $PWD"
# ./build_detect_lib.sh
echo "Lib built in ../libs/librknnyolo"
echo "Build my_rga_crop_rknn"
$GCC_PATH/arm-linux-gnueabihf-g++ -o build/rga_yolo my_rga_crop_rknn.cpp $PWD/../common/sample_common_isp.c \
-lrknnyolo \
-I$SDK_PATH/external/camera_engine_rkaiq/rkisp_demo/demo/libs/arm32/ \
-ldl -leasymedia -lrkaiq -lrkisp_api -lrockx -lrga \
-ljpeg -lpng -lrknn_api -lpthread -lz \
-I$SDK_PATH/external/rkmedia/include/rkmedia/ \
-I$SDK_PATH/external/rkmedia/examples/common/ \
-I$SDK_PATH/external/camera_engine_rkaiq/include/uAPI/ \
-I$SDK_PATH/external/camera_engine_rkaiq/include/xcore/ \
-I$SDK_PATH/external/camera_engine_rkaiq/include/algos/ \
-I$SDK_PATH/external/camera_engine_rkaiq/include/iq_parser/ \
-I$SDK_PATH/external/camera_engine_rkaiq/include/common/ \
-L$SDK_PATH/buildroot/output/rockchip_rv1126_rv1109/oem/usr/lib \
-L./libs \
-L./yolo_detect/libs/RV1109_1126/librknn_api/lib \
-L./yolo_detect/libs/RV1109_1126/zlib/lib \
-L./yolo_detect/libs/RV1109_1126/libpng/lib \
-L./yolo_detect/libs/RV1109_1126/libjpeg/lib \
-L$SYS_ROOT/usr/lib/ \
--sysroot=$SYS_ROOT
echo "Build end, the application is build/rga_yolo"
