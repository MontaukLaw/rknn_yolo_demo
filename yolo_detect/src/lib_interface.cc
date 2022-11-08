/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <vector>
#include <string>

#define _BASETSD_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#undef cimg_display
#define cimg_display 0
#undef cimg_use_jpeg
#define cimg_use_jpeg 1
#undef cimg_use_png
#define cimg_use_png 1
#include "CImg/CImg.h"

#include "drm_func.h"
#include "rga_func.h"
#include "rknn_api.h"
#include "postprocess.h"

#define PERF_WITH_POST 1
#define COCO_IMG_NUMBER 5000
#define YOLO_INPUT_DATASIZE 1228800  // 640*640*3
using namespace cimg_library;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
// Print RKNN tensor info like shape, type, quantization, etc.
static void printRKNNTensor(rknn_tensor_attr* attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
        "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
        attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
        attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
        attr->qnt_type, attr->fl, attr->zp, attr->scale);
}
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Load data from file, just like inputstream in java
static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

// Load image from file by file name
static unsigned char* load_model(const char* filename, int* model_size)
{

    FILE* fp;
    unsigned char* data;

    // Read model file
    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
    FILE* fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

static unsigned char* load_image_optimized(const char* image_path, int* org_height, int* org_width, int* org_ch, rknn_tensor_attr* input_attr)
{
    struct timeval start_time, stop_time;
    memset(&start_time, 0, sizeof(start_time));
    memset(&stop_time, 0, sizeof(stop_time));

    gettimeofday(&start_time, NULL);
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    req_height = input_attr->dims[1];
    req_width = input_attr->dims[0];
    req_channel = input_attr->dims[2];

    // printf("w=%d,h=%d,c=%d, fmt=%d\n", req_width, req_height, req_channel, input_attr->fmt);

    int height = 0;
    int width = 0;
    int channel = 0;

    // Use image lib to load image as char*
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image-%s failed!\n", image_path);
        return NULL;
    }

    *org_width = width;
    *org_height = height;
    *org_ch = channel;

    gettimeofday(&stop_time, NULL);
    printf("load_image_optimized %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // image_data is a char * like java outputstream, but it's RGB format
    return image_data;
}

// Load image from file by file name, and trans to NCHW format as model input attribute is NCHW
static unsigned char* load_image(const char* image_path, int* org_height, int* org_width, int* org_ch, rknn_tensor_attr* input_attr)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:

        req_height = input_attr->dims[2];
        req_width = input_attr->dims[1];
        req_channel = input_attr->dims[0];
        break;
    case RKNN_TENSOR_NCHW:
        // This way
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[0];
        req_channel = input_attr->dims[2];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    // printf("w=%d,h=%d,c=%d, fmt=%d\n", req_width, req_height, req_channel, input_attr->fmt);

    int height = 0;
    int width = 0;
    int channel = 0;

    // Use image lib to load image as char*
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image-%s failed!\n", image_path);
        return NULL;
    }

    // If it's a gray pic, only 1 channel, extend to RGB
    if (channel == 1) {
        printf("image is grey, convert to RGB");
        void* rgb_data = malloc(width * height * 3);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int offset = (i * width + j) * 3;
                ((unsigned char*)rgb_data)[offset] = ((unsigned char*)image_data)[offset];
                ((unsigned char*)rgb_data)[offset + 1] = ((unsigned char*)image_data)[offset];
                ((unsigned char*)rgb_data)[offset + 2] = ((unsigned char*)image_data)[offset];
            }
        }
        free(image_data);
        image_data = (unsigned char*)rgb_data;
        channel = 3;
    }

    // If image width size is not stride 4, resize it
    if (width % 4 != 0) {
        int new_width = width + (4 - width % 4);
        printf("%d is not pixel align, resize to %d, this will make the result shift slightly\n", width, new_width);
        void* resize_data = malloc(new_width * height * channel);
        stbir_resize_uint8(image_data, width, height, 0, (unsigned char*)resize_data, new_width, height, 0, channel);
        free(image_data);
        image_data = (unsigned char*)resize_data;
        *org_width = new_width;
    }
    else {
        *org_width = width;
    }

    *org_height = height;
    *org_ch = channel;

    // image_data is a char * like java outputstream, but it's RGB format
    return image_data;
}

void lib_test(void) {
    printf("here we are\n");
}

int input_img_height = 0;
int input_img_width = 0;
rga_context rga_ctx;
drm_context drm_ctx;
rknn_context ctx;
size_t actual_size = 0;
const float nms_threshold = 0.65;
const float conf_threshold = 0.6;   // As request change to 0.8 0.2;
unsigned int handle;
int input_channel = 3;
int model_input_width = 0;
int model_input_height = 0;
rknn_input inputs[1];
rknn_input_output_num io_num = { 0 ,0 };
rknn_output outputs[3];
rknn_tensor_attr output_attrs[3];
unsigned char* model_data = nullptr;
void* resize_buf = nullptr;
rknn_tensor_attr input_attrs[1];
int drm_fd = -1;
int img_channel = 0;
void* drm_buf = NULL;
int buf_fd = -1; // converted from buffer handle
char* image_name = (char*)"/mnt/tmp/input.jpg";
unsigned char* input_data = NULL;

int model_init(char* model_name, int inputImgHeight, int inputImgWidth) {

    printf("input_img_height=%d, input_img_width=%d\n", input_img_height, input_img_width);
    input_img_height = inputImgHeight;
    input_img_width = inputImgWidth;

    int status = 0;

    // converted from buffer handle

    int ret;

    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));

    printf("Loading model...\n");

    int model_data_size = 0;
    model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
        sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);


    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    memset(input_attrs, 0, sizeof(input_attrs));

    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
            sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    memset(output_attrs, 0, sizeof(output_attrs));

    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        printRKNNTensor(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_input_width = input_attrs[0].dims[0];
        model_input_height = input_attrs[0].dims[1];
        // So the width and height is 640, 640
    }
    else
    {
        printf("model is NHWC input fmt\n");
        model_input_width = input_attrs[0].dims[1];
        model_input_height = input_attrs[0].dims[2];
    }

    printf("model input height=%d, width=%d, channel=%d\n", model_input_height, model_input_width, input_channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_input_width * model_input_height * input_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;   // Default format is NHWC
    inputs[0].pass_through = 0;

    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

#if 0
    resize_buf = malloc(input_img_height * input_img_width * input_channel);
    if (resize_buf == nullptr) {
        printf("malloc resize_buf failed\n");
        return -1;
    }
#endif

    // DRM alloc buffer
    drm_fd = drm_init(&drm_ctx);

    // drm_buf_alloc
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, 640, 640, 3 * 8, &buf_fd, &handle, &actual_size);
    printf("drm_buf=%p, buf_fd=%d, handle=%d, actual_size=%d\n", drm_buf, buf_fd, handle, actual_size);
    return 0;

}

#if 0
// Load image from file by file name, and trans to NCHW format as model input attribute is NCHW
static unsigned char* data_to_image(int* org_height, int* org_width, int* org_ch, rknn_tensor_attr* input_attr)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;



    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:

        req_height = input_attr->dims[2];
        req_width = input_attr->dims[1];
        req_channel = input_attr->dims[0];
        break;
    case RKNN_TENSOR_NCHW:
        // This way
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[0];
        req_channel = input_attr->dims[2];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    // printf("w=%d,h=%d,c=%d, fmt=%d\n", req_width, req_height, req_channel, input_attr->fmt);

    int height = 0;
    int width = 0;
    int channel = 0;

    // Use image lib to load image as char*
    unsigned char* image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image-%s failed!\n", image_path);
        return NULL;
    }

    *org_width = width;
    *org_height = height;
    *org_ch = channel;

    // image_data is a char * like java outputstream, but it's RGB format
    return image_data;
}
#endif

void detect_adv(void) {

    struct timeval start_time, stop_time;
    memset(&start_time, 0, sizeof(start_time));
    memset(&stop_time, 0, sizeof(stop_time));

    // printf("model input h:%d w:%d\n", model_input_height, model_input_width);

    // start counting time
    gettimeofday(&start_time, NULL);
    int ret = 0;

    /* Input preprocess */
    // Load image
    // CImg<unsigned char> img_obj(image_name);

    input_data = load_image_optimized(image_name, &input_img_height, &input_img_width, &img_channel, &input_attrs[0]);
    if (!input_data)
    {
        return;
    }
    // printf("input_data=%p, input_img_height=%d, input_img_width=%d, img_channel=%d\n", input_data, input_img_height, input_img_width, img_channel);

    gettimeofday(&stop_time, NULL);
    printf("load image used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    gettimeofday(&start_time, NULL);

    memcpy(drm_buf, input_data, input_img_height * input_img_width * input_channel);

    // memset(resize_buf, 0, model_input_width * model_input_height * input_channel);

    gettimeofday(&stop_time, NULL);
    printf("DRM alloc buffer used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    gettimeofday(&start_time, NULL);

#if 0
    // Letter box resize
    float img_wh_ratio = (float)img_width / (float)img_height;
    float input_wh_ratio = (float)model_input_width / (float)model_input_height;
    float resize_scale = 0;
    int resize_width;
    int resize_height;
    int h_pad;
    int w_pad;
    if (img_wh_ratio >= input_wh_ratio) {
        //pad height dim
        resize_scale = (float)model_input_width / (float)img_width;
        resize_width = model_input_width;
        resize_height = (int)((float)img_height * resize_scale);
        w_pad = 0;
        h_pad = (model_input_height - resize_height) / 2;
    }
    else {
        //pad width dim
        resize_scale = (float)model_input_height / (float)img_height;
        resize_width = (int)((float)img_width * resize_scale);
        resize_height = model_input_height;
        w_pad = (model_input_width - resize_width) / 2;;
        h_pad = 0;
    }

    gettimeofday(&stop_time, NULL);
    printf("count pad %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0);
    gettimeofday(&start_time, NULL);

#endif
    // printf("w_pad: %d, h_pad: %d\n", w_pad, h_pad);
    // printf("resize_width: %d, resize_height: %d\n", resize_width, resize_height);

    // Init rga context
    // RGA_init(&rga_ctx);
    // img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);

    // inputs[0].buf = resize_buf;

    inputs[0].buf = drm_buf;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    /* Post process */
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;

    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    // gettimeofday(&stop_time, NULL);
    // printf("resize pic%f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0);
    // gettimeofday(&start_time, NULL);
    // printf("h_pad :%d, w_pad :%d, resize_scale:%f \n", h_pad, w_pad, resize_scale);

    post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
        0, 0, 1.0f, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);

    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        printf("\n");
        printf("----------->>>>>>>> found target label @: %s\n", det_result->name);
        printf("(%d %d %d %d) %f\n",
            det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
            det_result->prop);

    }

    gettimeofday(&stop_time, NULL);
    printf("detect once used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

#if 0
    // Draw Objects
    const unsigned char blue[] = { 0, 0, 255 };
    char score_result[64];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        printf("label @: %s\n", det_result->name);
        printf("(%d %d %d %d) %f\n",
            det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
            det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        int ret = snprintf(score_result, sizeof score_result, "%f", det_result->prop);
        //draw box
        img_obj.draw_rectangle(x1, y1, x2, y2, blue, 1, ~0U);
        img_obj.draw_text(x1, y1 - 24, det_result->name, blue);
        img_obj.draw_text(x1, y1 - 12, score_result, blue);
    }

    img_obj.save("/mnt/tmp/out.bmp");
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    gettimeofday(&stop_time, NULL);
    printf("draw write output %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);


    // loop test
    int test_count = 50;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i)
    {
        img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
            h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }

    gettimeofday(&stop_time, NULL);
    printf("run loop count = %d , average time: %f ms\n", test_count,
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);
#endif

    // drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    // drm_deinit(&drm_ctx, drm_fd);
    if (input_data) {
        free(input_data);
    }

    return;
}

#if 0
void detect_test(void) {

    struct timeval start_time, stop_time;
    memset(&start_time, 0, sizeof(start_time));
    memset(&stop_time, 0, sizeof(stop_time));

    void* drm_buf = NULL;
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle

    unsigned char* input_data = NULL;
    // printf("model input h:%d w:%d\n", model_input_height, model_input_width);

    // start counting time
    gettimeofday(&start_time, NULL);
    int ret = 0;
    int img_height = 0;
    int img_width = 0;
    int img_channel = 0;

    /* Input preprocess */
    // Load image
    // CImg<unsigned char> img_obj(image_name);

    input_data = load_image(image_name, &img_height, &img_width, &img_channel, &input_attrs[0]);
    if (!input_data)
    {
        return;
    }

    gettimeofday(&stop_time, NULL);
    printf("load image used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    gettimeofday(&start_time, NULL);

    // DRM alloc buffer
    drm_fd = drm_init(&drm_ctx);
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, img_width, img_height, input_channel * 8, &buf_fd, &handle, &actual_size);
    printf("drm_buf = %p, buf_fd = %d, handle = %d, actual_size = %d\n", drm_buf, buf_fd, handle, actual_size);

    memcpy(drm_buf, input_data, img_width * img_height * input_channel);
    memset(resize_buf, 0, model_input_width * model_input_height * input_channel);

    gettimeofday(&stop_time, NULL);
    printf("DRM alloc buffer used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    gettimeofday(&start_time, NULL);

#if 1
    // Letter box resize
    float img_wh_ratio = (float)img_width / (float)img_height;
    float input_wh_ratio = (float)model_input_width / (float)model_input_height;
    float resize_scale = 0;
    int resize_width;
    int resize_height;
    int h_pad;
    int w_pad;
    if (img_wh_ratio >= input_wh_ratio) {
        //pad height dim
        resize_scale = (float)model_input_width / (float)img_width;
        resize_width = model_input_width;
        resize_height = (int)((float)img_height * resize_scale);
        w_pad = 0;
        h_pad = (model_input_height - resize_height) / 2;
    }
    else {
        //pad width dim
        resize_scale = (float)model_input_height / (float)img_height;
        resize_width = (int)((float)img_width * resize_scale);
        resize_height = model_input_height;
        w_pad = (model_input_width - resize_width) / 2;;
        h_pad = 0;
    }

    gettimeofday(&stop_time, NULL);
    printf("count pad %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0);
    gettimeofday(&start_time, NULL);

#endif
    // printf("w_pad: %d, h_pad: %d\n", w_pad, h_pad);
    // printf("resize_width: %d, resize_height: %d\n", resize_width, resize_height);

    // Init rga context
    RGA_init(&rga_ctx);
    img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);

    inputs[0].buf = resize_buf;

    // inputs[0].buf = drm_buf;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    /* Post process */
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;

    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    // gettimeofday(&stop_time, NULL);
    // printf("resize pic%f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000.0);
    // gettimeofday(&start_time, NULL);
    printf("h_pad :%d, w_pad :%d, resize_scale:%f \n", h_pad, w_pad, resize_scale);

    post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
        h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);

    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        printf("->>>>>>>> found target label @: %s\n", det_result->name);
        printf("(%d %d %d %d) %f\n",
            det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
            det_result->prop);
    }

    gettimeofday(&stop_time, NULL);
    printf("detect once used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

#if 0
    // Draw Objects
    const unsigned char blue[] = { 0, 0, 255 };
    char score_result[64];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        printf("label @: %s\n", det_result->name);
        printf("(%d %d %d %d) %f\n",
            det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
            det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        int ret = snprintf(score_result, sizeof score_result, "%f", det_result->prop);
        //draw box
        img_obj.draw_rectangle(x1, y1, x2, y2, blue, 1, ~0U);
        img_obj.draw_text(x1, y1 - 24, det_result->name, blue);
        img_obj.draw_text(x1, y1 - 12, score_result, blue);
    }

    img_obj.save("/mnt/tmp/out.bmp");
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    gettimeofday(&stop_time, NULL);
    printf("draw write output %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);


    // loop test
    int test_count = 50;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i)
    {
        img_resize_slow(&rga_ctx, drm_buf, img_width, img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
            h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }

    gettimeofday(&stop_time, NULL);
    printf("run loop count = %d , average time: %f ms\n", test_count,
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);
#endif

    // drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    // drm_deinit(&drm_ctx, drm_fd);
    free(input_data);
    return;
}
#endif

int detect(unsigned char* data, int data_size) {return 0;}

int detect_by_buf(void* data) {
    int ret = 0;
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);

    // memcpy(drm_buf, data, YOLO_INPUT_DATASIZE);
    // printf("Data been copied, size is %d\n", data_size);

    inputs[0].buf = data; // drm_buf;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    /* Post process */
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;

    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf,
        (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
        0, 0, 1.0f, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);

    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        printf("\n");
        printf("----------->>>>>>>> found target label @: %s\n", det_result->name);
        printf("(%d %d %d %d) %f\n",
            det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
            det_result->prop);

    }

    gettimeofday(&stop_time, NULL);
    
    printf("whole detect used %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    return 0;
}

#if 0
int detect(unsigned char* data, int data_size) {
    struct timeval start_time, stop_time;
    int ret;
    int drm_fd = -1;
    void* drm_buf = NULL;

    unsigned char* input_data = NULL;

    // Get input input_img_height/width after stride align
    input_data = (unsigned char*)data;
    if (!input_data)
    {
        printf("input data error\n");
        return -1;
    }

    // DRM alloc buffer
    drm_fd = drm_init(&drm_ctx);
    drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, input_img_width, input_img_height, input_channel * 8, &buf_fd, &handle, &actual_size);
    memcpy(drm_buf, input_data, input_img_width * input_img_height * input_channel);
    memset(resize_buf, 0, model_input_width * model_input_height * input_channel);

    // Letter box resize
    // We need a 640*640 input, so we need to resize the image
    // model_input_width here mean model input width, 640, so the model_input_wh_ration whill be 1.0f
    float input_img_wh_ratio = (float)input_img_width / (float)input_img_height;
    float model_input_wh_ratio = (float)model_input_width / (float)model_input_height;
    float resize_scale = 0;
    int resize_width;
    int resize_height;
    int h_pad;
    int w_pad;

    // If the input_img_wh_ratio is bigger than model_input_wh_ratio(1.0f), it means it's a fat image
    if (input_img_wh_ratio >= model_input_wh_ratio) {
        // pad height dim
        // add some gray bar on the top and bottom of image
        resize_scale = (float)model_input_width / (float)input_img_width;
        resize_width = model_input_width;
        resize_height = (int)((float)input_img_height * resize_scale);
        w_pad = 0;
        h_pad = (model_input_height - resize_height) / 2;  // top and bottom pad size
    }
    else {  // it's tall image
        // pad width dim
        // add some gray area on the left and right side of image
        resize_scale = (float)model_input_height / (float)input_img_height;
        resize_width = (int)((float)input_img_width * resize_scale);
        resize_height = model_input_height;
        w_pad = (model_input_width - resize_width) / 2;   // left and right pad size
        h_pad = 0;
    }

    // printf("w_pad: %d, h_pad: %d\n", w_pad, h_pad);
    // printf("resize_width: %d, resize_height: %d\n", resize_width, resize_height);

    // Init rga context
    RGA_init(&rga_ctx);
    // resize the image, thrink it to 640*640
    img_resize_slow(&rga_ctx, drm_buf, input_img_width, input_img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
    inputs[0].buf = resize_buf;
    // Prepare to count time
    gettimeofday(&start_time, NULL);

    // Reload!
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    // Shoot!
    ret = rknn_run(ctx, NULL);

    /* Post process */
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<uint8_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
        h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);

    // Shougong!
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // Draw Objects
    // In our case , no need draw an object, but just need output a gpio change.
    printf("detect_result_group.count: %d\n", detect_result_group.count);

    const unsigned char blue[] = { 0, 0, 255 };
    char score_result[64];

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    // loop test
    int test_count = 50;
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i)
    {
        img_resize_slow(&rga_ctx, drm_buf, input_img_width, input_img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
        rknn_inputs_set(ctx, io_num.n_input, inputs);
        ret = rknn_run(ctx, NULL);
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
            h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    }

    gettimeofday(&stop_time, NULL);
    printf("run loop count = %d , average time: %f ms\n", test_count,
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

    return 0;

    drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    drm_deinit(&drm_ctx, drm_fd);
    free(input_data);

    drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
    drm_deinit(&drm_ctx, drm_fd);
    free(input_data);

    // release
    ret = rknn_destroy(ctx);

    RGA_deinit(&rga_ctx);
    if (model_data)
    {
        free(model_data);
    }

    if (resize_buf)
    {
        free(resize_buf);
    }

    return 0;
}
#endif

#if 0
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main_main(int argc, char** argv)
{
    int status = 0;
    char* model_name = NULL;
    rknn_context ctx;
    void* drm_buf = NULL;
    int drm_fd = -1;
    int buf_fd = -1; // converted from buffer handle
    unsigned int handle;
    size_t actual_size = 0;
    int input_img_width = 0;
    int input_img_height = 0;
    int img_channel = 0;
    rga_context rga_ctx;
    drm_context drm_ctx;
    const float nms_threshold = 0.65;
    const float conf_threshold = 0.8;   // As request change to 0.8 0.2;
    struct timeval start_time, stop_time;
    int ret;
    memset(&rga_ctx, 0, sizeof(rga_context));
    memset(&drm_ctx, 0, sizeof(drm_context));

    if (argc != 5)
    {
        printf("Usage: %s <rknn model> [fp/u8] [single_img/multi_imgs] <path>\n", argv[0]);
        return -1;
    }

    model_name = (char*)argv[1];
    char* post_process_type = argv[2];
    char* input_type = argv[3];
    char* input_path = argv[4];

    if (strcmp(post_process_type, "fp") == 0) {
        printf("Post process with fp\n");
    }
    else if (strcmp(post_process_type, "u8") == 0) {
        printf("Post process with u8\n");
    }
    else {
        printf("Post process type not support: %s\nPlease select from [fp/u8]\n", post_process_type);
        return -1;
    }

    if (strcmp(input_type, "single_img") == 0) {
        printf("Test with single img\n");
    }
    else if (strcmp(input_type, "multi_imgs") == 0) {
        printf("Test with multi imgs\n");
    }
    else {
        printf("Test input type is not support: %s\nPlease select from [single_img/multi_imgs]");
    }

    /* Create the neural network */
    printf("Loading model...\n");
    /* Load model from file*/
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_name, &model_data_size);
    /* Create rknn context, init rknn by model data*/
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    /* Query sdk version */
    /* Just confirm you already got RKNN lib on your board */
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
        sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    /* Get input,output attr */
    /* Get model input and output attribute */
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    /* Most like print will like input 1, output 3*/
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));

    /* Get input attribute by querrying from rknn*/
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
            sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }
    /*
        The input attribute will like
        index=0 name=images_165 n_dims=4 dims=[1 3 640 640] n_elems=1228800 size=1228800 fmt=0 type=3 qnt_type=2 fl=0 zp=0 scale=0.003922
    */

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        printRKNNTensor(&(output_attrs[i]));
    }
    /*
        The output attribute will like
        index=0 name=Conv_Conv_159/out0_0 n_dims=4 dims=[1 255 80 80] n_elems=1632000 size=1632000 fmt=0 type=3 qnt_type=2 fl=-69 zp=187 scale=0.127843
        index=1 name=Conv_Conv_160/out0_1 n_dims=4 dims=[1 255 40 40] n_elems=408000 size=408000 fmt=0 type=3 qnt_type=2 fl=-74 zp=182 scale=0.113217
        index=2 name=Conv_Conv_161/out0_2 n_dims=4 dims=[1 255 20 20] n_elems=102000 size=102000 fmt=0 type=3 qnt_type=2 fl=-84 zp=172 scale=0.103272
        model is NCHW input fmt
    */

    // We have 3 input channels, RGB
    int input_channel = 3;
    int model_input_width = 0;
    int model_input_height = 0;

    // NCHW mean N: batch, C: channel, H: height, W: width
    // Check the input attribute print result, the [1 3 640 640] actually is [dims[3] dims[2] dims[1] dims[0]]
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        model_input_width = input_attrs[0].dims[0];
        model_input_height = input_attrs[0].dims[1];
        // So the width and height is 640, 640
    }
    else
    {
        printf("model is NHWC input fmt\n");
        model_input_width = input_attrs[0].dims[1];
        model_input_height = input_attrs[0].dims[2];
    }

    // As you already know
    printf("model input height=%d, width=%d, channel=%d\n", model_input_height, model_input_width, input_channel);

    /* Init input tensor */
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_input_width * model_input_height * input_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;   // Default format is NHWC
    inputs[0].pass_through = 0;

    /* Init output tensor */
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        // FP slower than UINT8
        if (strcmp(post_process_type, "fp") == 0) {
            outputs[i].want_float = 1;
        }
        else if (strcmp(post_process_type, "u8") == 0) {
            outputs[i].want_float = 0;
        }
    }

    void* resize_buf = malloc(model_input_height * model_input_width * input_channel);
    unsigned char* input_data = NULL;

    /* Single img input */
    /* Due to different input img size, multi img method has to calculate letterbox param each time*/
    if (strcmp(input_type, "single_img") == 0)
    {
        char* image_name = input_path;

        /* Input preprocess */
        // Load image
        CImg<unsigned char> img(image_name);

        // Get input input_img_height/width after stride align
        input_data = load_image(image_name, &input_img_height, &input_img_width, &img_channel, &input_attrs[0]);
        if (!input_data)
        {
            return -1;
        }

        // DRM alloc buffer
        drm_fd = drm_init(&drm_ctx);
        drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, input_img_width, input_img_height, input_channel * 8,
            &buf_fd, &handle, &actual_size);
        memcpy(drm_buf, input_data, input_img_width * input_img_height * input_channel);
        memset(resize_buf, 0, model_input_width * model_input_height * input_channel);

        // Letter box resize
        // We need a 640*640 input, so we need to resize the image
        // model_input_width here mean model input width, 640, so the model_input_wh_ration whill be 1.0f
        float input_img_wh_ratio = (float)input_img_width / (float)input_img_height;
        float model_input_wh_ratio = (float)model_input_width / (float)model_input_height;
        float resize_scale = 0;
        int resize_width;
        int resize_height;
        int h_pad;
        int w_pad;

        // If the input_img_wh_ratio is bigger than model_input_wh_ratio(1.0f), it means it's a fat image
        if (input_img_wh_ratio >= model_input_wh_ratio) {
            // pad height dim
            // add some gray bar on the top and bottom of image
            resize_scale = (float)model_input_width / (float)input_img_width;
            resize_width = model_input_width;
            resize_height = (int)((float)input_img_height * resize_scale);
            w_pad = 0;
            h_pad = (model_input_height - resize_height) / 2;  // top and bottom pad size
        }
        else {  // it's tall image
            // pad width dim
            // add some gray area on the left and right side of image
            resize_scale = (float)model_input_height / (float)input_img_height;
            resize_width = (int)((float)input_img_width * resize_scale);
            resize_height = model_input_height;
            w_pad = (model_input_width - resize_width) / 2;   // left and right pad size
            h_pad = 0;
        }

        // printf("w_pad: %d, h_pad: %d\n", w_pad, h_pad);
        // printf("resize_width: %d, resize_height: %d\n", resize_width, resize_height);

        // Init rga context
        RGA_init(&rga_ctx);
        // resize the image, thrink it to 640*640
        img_resize_slow(&rga_ctx, drm_buf, input_img_width, input_img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
        inputs[0].buf = resize_buf;
        // Prepare to count time
        gettimeofday(&start_time, NULL);

        // Reload!
        rknn_inputs_set(ctx, io_num.n_input, inputs);

        // Shoot!
        ret = rknn_run(ctx, NULL);

        // Bang!
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        /* Post process */
        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<uint8_t> out_zps;
        for (int i = 0; i < io_num.n_output; ++i)
        {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }

        if (strcmp(post_process_type, "u8") == 0) {
            post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
                h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);
        }
        else if (strcmp(post_process_type, "fp") == 0) {
            post_process_fp((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, model_input_height, model_input_width,
                h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, &detect_result_group);
        }

        // Shougong!
        gettimeofday(&stop_time, NULL);
        printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

        // Draw Objects
        // In our case , no need draw an object, but just need output a gpio change.
        printf("detect_result_group.count: %d\n", detect_result_group.count);

        goto RELEASE_RESOURCE;

        const unsigned char blue[] = { 0, 0, 255 };
        char score_result[64];
        // Draw result frame one by one
        for (int i = 0; i < detect_result_group.count; i++)
        {
            detect_result_t* det_result = &(detect_result_group.results[i]);

            printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left,
                det_result->box.top, det_result->box.right, det_result->box.bottom, det_result->prop);

            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            int ret = snprintf(score_result, sizeof score_result, "%f", det_result->prop);
            //draw box
            img.draw_rectangle(x1, y1, x2, y2, blue, 1, ~0U);
            img.draw_text(x1, y1 - 24, det_result->name, blue);
            img.draw_text(x1, y1 - 12, score_result, blue);
        }
        img.save("./out.bmp");
        ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

        // loop test
        int test_count = 50;
        gettimeofday(&start_time, NULL);
        for (int i = 0; i < test_count; ++i)
        {
            img_resize_slow(&rga_ctx, drm_buf, input_img_width, input_img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
            rknn_inputs_set(ctx, io_num.n_input, inputs);
            ret = rknn_run(ctx, NULL);
            ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

#if (PERF_WITH_POST == 1)
            if (strcmp(post_process_type, "u8") == 0) {
                post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
                    h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);
            }
            else if (strcmp(post_process_type, "fp") == 0) {
                post_process_fp((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, model_input_height, model_input_width,
                    h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, &detect_result_group);
            }
#endif
            ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
        }

        gettimeofday(&stop_time, NULL);
        printf("run loop count = %d , average time: %f ms\n", test_count,
            (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

        drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
        drm_deinit(&drm_ctx, drm_fd);
        free(input_data);
    }
    /* multi imgs input */
    /* Usually for coco benchmark*/
    else if (strcmp(input_type, "multi_imgs") == 0)
    {
        FILE* output_file = NULL;
        output_file = fopen("./result_record.txt", "w+");

        char* img_paths[COCO_IMG_NUMBER];
        ret = readLines(input_path, img_paths, COCO_IMG_NUMBER);

        drm_fd = drm_init(&drm_ctx);
        RGA_init(&rga_ctx);

        for (int j = 0; j < COCO_IMG_NUMBER; j++)
        {
            printf("[%d/%d]Detect on %s\n", j + 1, COCO_IMG_NUMBER, img_paths[j]);
            /* Input preprocess */
            // Load image
            CImg<unsigned char> img(img_paths[j]);
            input_data = load_image(img_paths[j], &input_img_height, &input_img_width, &img_channel, &input_attrs[0]);
            if (!input_data)
            {
                return -1;
            }

            // DRM alloc buffer
            drm_buf = drm_buf_alloc(&drm_ctx, drm_fd, input_img_width, input_img_height, input_channel * 8,
                &buf_fd, &handle, &actual_size);
            memcpy(drm_buf, input_data, input_img_width * input_img_height * input_channel);
            memset(resize_buf, 0, model_input_width * model_input_height * input_channel);

            // Letter box resize
            float input_img_wh_ratio = (float)input_img_width / (float)input_img_height;
            float model_input_wh_ratio = (float)model_input_width / (float)model_input_height;
            float resize_scale = 0;
            int resize_width;
            int resize_height;
            int h_pad;
            int w_pad;
            if (input_img_wh_ratio >= model_input_wh_ratio) {
                //pad height dim
                resize_scale = (float)model_input_width / (float)input_img_width;
                resize_width = model_input_width;
                resize_height = (int)((float)input_img_height * resize_scale);
                w_pad = 0;
                h_pad = (model_input_height - resize_height) / 2;
            }
            else {
                //pad width dim
                resize_scale = (float)model_input_height / (float)input_img_height;
                resize_width = (int)((float)input_img_width * resize_scale);
                resize_height = model_input_height;
                w_pad = (model_input_width - resize_width) / 2;;
                h_pad = 0;
            }

            // Init rga context
            img_resize_slow(&rga_ctx, drm_buf, input_img_width, input_img_height, resize_buf, resize_width, resize_height, w_pad, h_pad);
            inputs[0].buf = resize_buf;
            gettimeofday(&start_time, NULL);
            rknn_inputs_set(ctx, io_num.n_input, inputs);

            ret = rknn_run(ctx, NULL);
            ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

            /* Post process */
            detect_result_group_t detect_result_group;
            std::vector<float> out_scales;
            std::vector<uint8_t> out_zps;
            for (int i = 0; i < io_num.n_output; ++i)
            {
                out_scales.push_back(output_attrs[i].scale);
                out_zps.push_back(output_attrs[i].zp);
            }
            // Get 3 dims output
            if (strcmp(post_process_type, "u8") == 0) {
                post_process_u8((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, model_input_height, model_input_width,
                    h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, out_zps, out_scales, &detect_result_group);
            }
            else if (strcmp(post_process_type, "fp") == 0) {
                post_process_fp((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, model_input_height, model_input_width,
                    h_pad, w_pad, resize_scale, conf_threshold, nms_threshold, &detect_result_group);
            }

            gettimeofday(&stop_time, NULL);
            printf("once run use %f ms\n",
                (__get_us(stop_time) - __get_us(start_time)) / 1000);

            // Draw Objects
            const unsigned char blue[] = { 0, 0, 255 };
            char score_result[64];
            for (int i = 0; i < detect_result_group.count; i++)
            {
                detect_result_t* det_result = &(detect_result_group.results[i]);
                printf("%s @ (%d %d %d %d) %f\n",
                    det_result->name,
                    det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                    det_result->prop);
                fprintf(output_file, "%s,%s,%d,%f,(%d %d %d %d)\n",
                    img_paths[j],
                    det_result->name,
                    det_result->class_index,
                    det_result->prop,
                    det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom);
            }
            ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

            drm_buf_destroy(&drm_ctx, drm_fd, buf_fd, handle, drm_buf, actual_size);
            free(input_data);
        }
        fclose(output_file);
        drm_deinit(&drm_ctx, drm_fd);
    }

RELEASE_RESOURCE:

    // release
    ret = rknn_destroy(ctx);

    RGA_deinit(&rga_ctx);
    if (model_data)
    {
        free(model_data);
    }

    if (resize_buf)
    {
        free(resize_buf);
    }

    return 0;
    }

#endif
