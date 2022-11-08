extern "C" {
#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <rga/im2d.h>
#include <rga/rga.h>
#include "../common/sample_common.h"
    // #include "librtsp/rtsp_demo.h"
#include "rkmedia_api.h"
#include "rkmedia_venc.h"
}

#include "rknnyolo.h"

#define YOLO_MODEL_INPUT_PIC_STIDE 640
#define YOLO_INPUT_SIZE (YOLO_MODEL_INPUT_PIC_STIDE * YOLO_MODEL_INPUT_PIC_STIDE * 3)

static bool quit = false;
IMAGE_TYPE_E g_enPixFmt = IMAGE_TYPE_NV12;
static MPP_CHN_S stSrcChn;
static MPP_CHN_S stDestChn;
static VI_CHN viChn = 0;
static RGA_CHN rgaChn = 0;
RK_S32 s32CamId = 0;
char* pDeviceName = (char*)"rkispp_scale0";
char* pciqFileDir = (char*)"/oem/etc/iqfiles";
RK_U32 u32SrcWidth = 1920;
RK_U32 u32SrcHeight = 1080;
RK_BOOL bMutictx = RK_FALSE;
int ret = 0;
char saveFilePath[128];

void set_vi_chn(void) {
    VI_CHN_ATTR_S viChnAttr;
    memset(&viChnAttr, 0, sizeof(VI_CHN_ATTR_S));
    viChnAttr.pcVideoNode = (RK_CHAR*)pDeviceName;
    viChnAttr.u32BufCnt = 4;
    viChnAttr.u32Width = u32SrcWidth;
    viChnAttr.u32Height = u32SrcHeight;
    viChnAttr.enPixFmt = g_enPixFmt;
    viChnAttr.enBufType = VI_CHN_BUF_TYPE_MMAP;
    viChnAttr.enWorkMode = VI_WORK_MODE_NORMAL;

    ret = RK_MPI_VI_SetChnAttr(s32CamId, viChn, &viChnAttr);
    ret |= RK_MPI_VI_EnableChn(s32CamId, viChn);

    if (ret) {
        printf("Create vi failed! ret=%d\n", ret);
    }
}

int bind_rga(void) {
    RGA_ATTR_S stRgaAttr;
    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = 2;
    stRgaAttr.u16Rotaion = 180;

    stRgaAttr.stImgIn.u32X = (1920 - 640) / 2;
    stRgaAttr.stImgIn.u32Y = (1080 - 640) / 2;
    stRgaAttr.stImgIn.imgType = IMAGE_TYPE_NV12;
    stRgaAttr.stImgIn.u32Width = 640;
    stRgaAttr.stImgIn.u32Height = 640;
    stRgaAttr.stImgIn.u32HorStride = 1920;
    stRgaAttr.stImgIn.u32VirStride = 1080;

    stRgaAttr.stImgOut.u32X = 0;  // (1920 - 640) / 2;
    stRgaAttr.stImgOut.u32Y = 0;  // (1080 - 640) / 2;
    stRgaAttr.stImgOut.imgType = IMAGE_TYPE_NV12;
    stRgaAttr.stImgOut.u32Width = 640;
    stRgaAttr.stImgOut.u32Height = 640;
    stRgaAttr.stImgOut.u32HorStride = 640;
    stRgaAttr.stImgOut.u32VirStride = 640;

    ret = RK_MPI_RGA_CreateChn(rgaChn, &stRgaAttr);
    if (ret) {
        printf("ERROR: Create rga[0] falied! ret=%d\n", ret);
        return -1;
    }

    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = s32CamId;
    stSrcChn.s32ChnId = viChn;

    stDestChn.enModId = RK_ID_RGA;
    stDestChn.s32ChnId = rgaChn;

    ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);
    if (ret) {
        printf("ERROR: Bind vi[0] and rga[0] failed! ret=%d\n", ret);
        return -1;
    }

    printf("RGA version: %s \n", querystring(RGA_VERSION));

    return 0;

}

static void sigterm_handler(int sig) {
    fprintf(stderr, "signal %d\n", sig);
    quit = true;
}

static void write_mb_file(MEDIA_BUFFER mb, int cnt) {
    sprintf(saveFilePath, "/tmp/rga_%d.rgb", cnt);
    FILE* save_file = fopen(saveFilePath, "w");
    if (!save_file) {
        printf("ERROR: Open %s failed!\n", saveFilePath);
    }
    fwrite(RK_MPI_MB_GetPtr(mb), 1, RK_MPI_MB_GetSize(mb), save_file);
    fclose(save_file);
}

// use rga to transform nv12 to rgb
static int nv12_to_rgb24_640x640(void* yuvBuffer, void* rgbBuffer) {

    rga_buffer_t src, dst;
    memset(&src, 0, sizeof(rga_buffer_t));
    memset(&dst, 0, sizeof(rga_buffer_t));

    src = wrapbuffer_virtualaddr(yuvBuffer, 640, 640, RK_FORMAT_YCbCr_420_SP);
    dst = wrapbuffer_virtualaddr(rgbBuffer, 640, 640, RK_FORMAT_RGB_888);

    src.format = RK_FORMAT_YCbCr_420_SP;
    dst.format = RK_FORMAT_RGB_888;

    IM_STATUS status = imcvtcolor(src, dst, src.format, dst.format);
    if (status != IM_STATUS_SUCCESS) {
        printf("ERROR: imcvtcolor failed!\n");
        return -1;
    }

    return 0;
}

static void write_rgb_file(void* data) {
    static int counter = 0;
    counter++;
    sprintf(saveFilePath, "/tmp/rga_%d.rgb", counter);
    FILE* save_file = fopen(saveFilePath, "w");
    if (!save_file) {
        printf("ERROR: Open %s failed!\n", saveFilePath);
    }
    fwrite(data, 1, YOLO_INPUT_SIZE, save_file);
    fclose(save_file);
}

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static void* get_stream_thread(void* arg) {

    VENC_RECV_PIC_PARAM_S stRecvParam;
    int cnt = 0;
    stRecvParam.s32RecvPicNum = 1;
    MEDIA_BUFFER srcMb = NULL;
    struct timeval start_time, stop_time;

    while (!quit) {
        gettimeofday(&start_time, NULL);
        // get stream from RGA
        srcMb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, rgaChn, -1);
        if (!srcMb) {
            printf("RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
            usleep(10000);
            break;
        }

        printf("#%d Get Frame:ptr:%p, size:%zu, mode:%d, channel:%d, "
            "timestamp:%lld\n",
            cnt++, RK_MPI_MB_GetPtr(srcMb), RK_MPI_MB_GetSize(srcMb),
            RK_MPI_MB_GetModeID(srcMb), RK_MPI_MB_GetChannelID(srcMb),
            RK_MPI_MB_GetTimestamp(srcMb));

        void* pRknnInputData = malloc(YOLO_INPUT_SIZE);
        ret = nv12_to_rgb24_640x640(RK_MPI_MB_GetPtr(srcMb), pRknnInputData);

        gettimeofday(&stop_time, NULL);
        printf("trans color used %f ms\n", (get_us(stop_time) - get_us(start_time)) / 1000.0);
        // actually, will cause 20ms

        if (ret == 0) {
            detect_by_buf(pRknnInputData);
            // write_rgb_file(pRknnInputData);
            // write_mb_file(pRknnInputData, cnt);
        }

        RK_MPI_MB_ReleaseBuffer(srcMb);

        free(pRknnInputData);

        // usleep(100000);  // 30ms
    }

    return NULL;
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("please input model name\n");
        return -1;
    }

    char* model_name = argv[1];

    ret = model_init(model_name, 640, 640);
    if (ret < 0) {
        printf("model init failed\n");
        return -1;
    }

    rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
    int fps = 10;
    SAMPLE_COMM_ISP_Init(s32CamId, hdr_mode, bMutictx, pciqFileDir);
    SAMPLE_COMM_ISP_Run(s32CamId);
    SAMPLE_COMM_ISP_SetFrameRate(s32CamId, fps);

    ret = RK_MPI_SYS_Init();
    if (ret) {
        printf("sys init failed ret=%d\n", ret);
        return -1;
    }

    RK_MPI_SYS_Init();

    set_vi_chn();


    bind_rga();

    pthread_t read_thread_id;
    pthread_create(&read_thread_id, NULL, get_stream_thread, NULL);

    usleep(1000); // waite for thread ready.

    ret = RK_MPI_VI_StartStream(s32CamId, viChn);
    if (ret) {
        printf("ERROR: Start Vi[0] failed! ret=%d\n", ret);
        return -1;
    }

    printf("%s initial finish\n", __func__);

    while (!quit) {
        usleep(500000);
    }

    RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
    RK_MPI_VENC_DestroyChn(0);
    RK_MPI_VI_DisableChn(s32CamId, 1);

    SAMPLE_COMM_ISP_Stop(s32CamId);

    return 0;


}
