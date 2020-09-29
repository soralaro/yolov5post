#ifndef CAFFE_YOLOV5_LAYER_HPP_
#define CAFFE_YOLOV5_LAYER_HPP_
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
//#include "json_config.h"

namespace Yolov5 {
    class YoloLayer {
#define sigmoid(a) (1/(1+exp(-(a))+0.0001));
    public:
        static const int LOCATIONS = 4;
        struct Detection{
            //cx cy w h
            float bbox[LOCATIONS];
            //float objectness;
            int classId;
            float prob;
		   float area;
        };

        YoloLayer(){
        }
        ~YoloLayer(){
        }
		void init(){
			//****************initialize the parameter*****************************/
			shape_w_=640;
			shape_h_=640;
			batch_size_=1;
			num_classes_=4;
			num_xywhc_=4+1+num_classes_;//x,y,w,h,obj_conf, class_conf.....
			class_names_={"person","car","bicycle","tricycle"};
			//3 vectors: layers_num: 3,anchors_num: 3,xy: 2
			anchors_={  { {10,13},{16,30},{33,23} },  { {30,61},{62,45},{59,119} }, { {116,90},{156,198},{373,326} }};
			num_anchors_=3;
			num_layers_=3;
			sample_rate_.resize(num_layers_);
			sample_rate_[0]=8;
			sample_rate_[1]=16;
			sample_rate_[2]=32;
			conf_thres_=0.4;
			iou_thres_=0.5;
			layer_width_={80,40,20};
			layer_height_={80,40,20};
			//***********************************************************************/

			//*********************calculate the parameter **************************/		
			xywhc_stride_.resize(num_layers_);
			anchors_stride_.resize(num_layers_);
			layer_stride_.resize(num_layers_);
			log_thres_=log(conf_thres_/(1-1.0001*conf_thres_));
			for(int i=0;i<num_layers_;i++){
				xywhc_stride_[i]= layer_width_[i]*layer_height_[i];
				anchors_stride_[i]=xywhc_stride_[i]*num_xywhc_;
				layer_stride_[i]=anchors_stride_[i]*num_anchors_;
			}
		}
		void decode(float *data,int size,std::vector<Detection> &vObj){
			float *y_data;
			float *x_data;
			float *layer_data=data;
			float *anchor_data=0;
			for(int l=0;l<num_layers_;l++){
				int s=xywhc_stride_[l];
				int w=layer_width_[l];
				int h=layer_height_[l];
				int anchors_size=anchors_[l].size();
				anchor_data=layer_data+4*s;//set data offset to object confidence
				for(int an=0;an<anchors_size;an++) {
					y_data=anchor_data; 
					for (int y = 0; y < h; y++) {
						x_data=y_data;
						for (int x = 0; x < w; x++) {
							if(*x_data>log_thres_){
								float obj_conf=sigmoid(x_data[0]);
								float *item=x_data-4*s; //set offset to begin
								for(int c=0;c<num_classes_;c++){
									float conf_c=sigmoid(item[(c+5) * s]);
									conf_c*=obj_conf;
									Detection obj;
									bool local_set=false;
									if(conf_c>conf_thres_){
										if(!local_set){
											float sig=sigmoid(item[0]);
											float cx = (sig * 2 - 0.5 + x) * sample_rate_[l]; //x
											sig=sigmoid(item[s]);
											float cy = (sig * 2 - 0.5 + y) * sample_rate_[l]; //y
											sig=sigmoid(item[2*s]);
											float w = sig * sig * 4 * anchors_[l][an][0];//w
											sig=sigmoid(item[3*s]);
											float h = sig * sig * 4  * anchors_[l][an][1];///h
											obj.bbox[0]=cx-w/2;
											obj.bbox[1]=cy-h/2;
											obj.bbox[2]=cx+w/2;
											obj.bbox[3]=cy+h/2;
											obj.area=(w+1)*(h+1);
											local_set=true;
										}
										obj.classId=c;
										obj.prob=conf_c;
										vObj.push_back(obj);
									}
								}
							}
							x_data++;
						}
						y_data+=layer_width_[l];
					}
					anchor_data+=anchors_stride_[l];
				}
				layer_data+=layer_stride_[l];
			}
		}

        float iouCompute(float * lbox, float* rbox,float lbox_area,float rbox_area) {
			float inter_lx=lbox[0]>rbox[0]?lbox[0]:rbox[0];
			float inter_ty=lbox[1]>rbox[1]?lbox[1]:rbox[1];
			float inter_rx=lbox[2]<rbox[2]?lbox[2]:rbox[2];
			float inter_by=lbox[3]<rbox[3]?lbox[3]:rbox[3];
			float inter_w=inter_rx-inter_lx;
			if(inter_w<=0)
				return 0;
			float inter_h=inter_by-inter_ty;
			if(inter_h<=0)
				return 0;
			float inter_area=(inter_w+1)*(inter_h+1);
			return inter_area/(lbox_area+rbox_area-inter_area);
        }

        void DoNms(std::vector<Detection>& detections) {
            std::vector<std::vector<Detection>> resClass;
            resClass.resize(num_classes_);

            for(unsigned int i = 0; i < detections.size(); ++i) {
                Detection item = detections[i];
                resClass[item.classId].push_back(item);
            }

            std::vector<Detection> result;
            for (unsigned int i = 0; i < num_classes_; ++i) {
                std::vector<Detection> dets = resClass[i];
                if(dets.size() == 0)
                    continue;

				std::sort(dets.begin(), dets.end(),
                     [](const Detection &left, const Detection &right) {
                         return left.prob > right.prob;
                     });
                //used for eval; each class keep number of objects at most
                if (top_k_ > -1 && top_k_ < (int)dets.size()) {
                    dets.resize(top_k_);
                }
                for (unsigned int m = 0; m < dets.size(); ++m) {
                    Detection item = dets[m];
                    result.push_back(item);
                    for(unsigned int n = m + 1; n < dets.size() ; ++n) {
                        if (iouCompute(item.bbox, dets[n].bbox,item.area,dets[n].area) > iou_thres_) {
                            dets.erase(dets.begin() + n);
                            --n;
                        }
                    }
                }
            }
            detections = result;
        }
        void postprocess(float *data,int size)
        {
            std::vector<Detection> vObj;
            decode(data, size,vObj);

            DoNms(vObj);
		   for(int i=0;i<vObj.size();i++)
		   {
			printf("lx=%f,ty=%f rx=%f, by=%f,class=%d,prob=%f \n",vObj[i].bbox[0],vObj[i].bbox[1],vObj[i].bbox[2],
					vObj[i].bbox[3],vObj[i].classId,vObj[i].prob);

		   }
		   if(vObj.empty())
		   {
			   printf("Detector no obj\n");
		   }


        }

    private:
       int shape_w_=640;
        int shape_h_=640;
        int batch_size_=1;
        int num_classes_=4;
        std::vector<std::string>class_names_={"person","car","bicycle","tricycle"};
        std::vector<std::vector<std::vector<int>>> anchors_={  { {10,13},{16,30},{33,23} },  { {30,61},{62,45},{59,119} }, { {116,90},{156,198},{373,326} }};
        std::vector<int> sample_rate_={8,16,32};
        int num_anchors_=3;
        int num_layers_=3;
        int num_xywhc_=0;
        float conf_thres_=0.4;
	    float log_thres_;
        float iou_thres_=0.5;
        std::vector<int>layer_width_;
        std::vector<int>layer_height_;
        std::vector<int>xywhc_stride_;
        std::vector<int>anchors_stride_;
        std::vector<int>layer_stride_;
        int top_k_=1000;
    };
}  // namespace caffe

#endif  // CAFFE_SLICE_LAYER_HPP_
