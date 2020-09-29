#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "yolov5_postprocess.hpp"
int main(int argc, char* argv[])
{
    std::fstream file("feature_map.bin", std::ios::in);
    file.seekg(0, std::ios::end);
    long file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    unsigned char *feature_map= new unsigned char[file_size];
    memset(feature_map,0,file_size);
    file.read(reinterpret_cast<char *>(feature_map), file_size);
    Yolov5::YoloLayer yolo;
    yolo.init();
    yolo.postprocess((float*)feature_map,file_size/sizeof(float));
}
