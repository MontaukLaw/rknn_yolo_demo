#ifndef __RKNN_YOLO_H_
#define __RKNN_YOLO_H_

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <climits>
#include <ctime>
#include <exception>
#include <algorithm>

void lib_test(void);

// int detect(void* data, int data_size, char* model_name, int input_img_height, int input_img_width);
int detect(unsigned char* data, int data_size);

int model_init(char* model_name, int inputImgHeight, int inputImgWidth);

// int detect_test(void);

void detect_adv(void);

int detect_by_buf(void* data);

#endif
