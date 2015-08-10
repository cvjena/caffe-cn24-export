#include <cstring>
#include <cstdlib>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <sstream>
#include <string>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <cn24.h>

using std::ofstream;

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    return 1;
  }
  Conv::System::Init();
  std::stringstream ss;

  NetParameter net_param;
  string input_filename(argv[1]);
  string output_filename(argv[2]);
  std::ofstream output_file(output_filename, std::ios::out | std::ios::binary);
  
  if (!ReadProtoFromBinaryFile(input_filename, &net_param)) {
    LOGERROR << "Failed to parse input binary file as NetParameter: "
               << input_filename;
    return 2;
  }

  LOGINFO << "Processing model: " << net_param.name();
  LOGINFO << "Layers in this model: " << net_param.layer_size();

  for(int layer_id = 0; layer_id < net_param.layer_size(); layer_id++) {
    const LayerParameter& layer_param = net_param.layer(layer_id);
    LOGINFO << "Processing layer " << layer_param.name();
    if(layer_param.type().compare("Convolution") == 0) {
      const ConvolutionParameter& cparam = layer_param.convolution_param();
      unsigned int kx = 0, ky = 0, sx = 1, sy = 1, px = 0, py = 0, kernels = 0, group = 1;

      // Parse kernel size
      if(cparam.has_kernel_size()) {
        kx = cparam.kernel_size(); ky = cparam.kernel_size();
      } else if(cparam.has_kernel_h() && cparam.has_kernel_w()) {
        kx = cparam.kernel_w(); ky = cparam.kernel_h();
      } else {
        FATAL("Kernel size missing");
      }

      // Parse padding
      if(cparam.has_pad()) {
        px = cparam.pad(); py = cparam.pad();
      } else if(cparam.has_pad_w() && cparam.has_pad_h()) {
        px = cparam.pad_w(); py = cparam.pad_h();
      } else {
        px = 0; py = 0;
      }

      // Parse stride
      if(cparam.has_stride()) {
        sx = cparam.stride(); sy = cparam.stride();
      } else if (cparam.has_stride_w() && cparam.has_stride_h()) {
        sx = cparam.stride_w(); sy = cparam.stride_h();
      } else {
        sx = 1; sy = 1;
      }

      // Parse group
      group = cparam.group();

      // Parse kernel count
      kernels = cparam.num_output();
      
      ss << "?convolutional size=" << kx << "x" << ky << " kernels=" << kernels;
      if((sx + sy) > 1)
        ss << " stride=" << sx << "x" << sy;
      if((px + py) > 0)
        ss << " pad=" << px << "x" << py;
      ss << " group=" << group;
      ss << "\n";
    } 
    else if(layer_param.type().compare("LRN") == 0) {
      const LRNParameter& lparam = layer_param.lrn_param();
      Conv::LocalResponseNormalizationLayer::NormalizationMethod method;
      float alpha = 1.0, beta = 0.75;
      unsigned int local_size = 5;

      ss << "?lrn";
      
      // Parse alpha and beta
      if(lparam.has_alpha()) {
        alpha = lparam.alpha();
      } 
      if(lparam.has_beta()) {
        beta = lparam.beta();
      } 
      ss << " alpha=" << alpha << " beta=" << beta;

      // Parse size
      if(lparam.has_local_size()) {
        local_size = lparam.local_size();
      }
      ss << " size=" << local_size;

      // Parse method
      if(lparam.has_norm_region()) {
        if(lparam.norm_region() == LRNParameter::ACROSS_CHANNELS) {
          method = Conv::LocalResponseNormalizationLayer::ACROSS_CHANNELS;
        } else if(lparam.norm_region() == LRNParameter::WITHIN_CHANNEL) {
          method = Conv::LocalResponseNormalizationLayer::WITHIN_CHANNELS;
        } else {
          FATAL("Unknown normalization method");
        }
      } else {
        method = Conv::LocalResponseNormalizationLayer::ACROSS_CHANNELS;
      }
      switch(method) {
        case Conv::LocalResponseNormalizationLayer::ACROSS_CHANNELS:
          ss << " method=across";
          break;
        case Conv::LocalResponseNormalizationLayer::WITHIN_CHANNELS:
          ss << " method=within";
          break;
      }

      ss << "\n";
    }
    else if(layer_param.type().compare("ReLU") == 0) {
      ss << "?relu\n";
    }
    else if(layer_param.type().compare("Data") == 0) {
      // Ignore data layer for now
    }
    else {
      LOGWARN << "Unknown layer type: " << layer_param.type();
    }

/*    for (int i = 0; i < layer_param.blobs_size(); ++i) {
      LOG(ERROR) << "Processing blob: " << i;
      Conv::Tensor tensor;
      Blob<float> blob;
      blob.FromProto(layer_param.blobs(i));
      LOG(ERROR) << "Shape: " << blob.shape(0) << "," << blob.shape(1) << "," << blob.shape(2) << "," << blob.shape(3);
      unsigned int width = blob.shape(2);
      unsigned int height = blob.shape(3);
      unsigned int maps = blob.shape(1);
      unsigned int samples = blob.shape(0);
      tensor.Resize(samples, width, height, maps);
      Conv::datum* target = tensor.data_ptr();
      float* source = (float*)blob.cpu_data();
      std::memcpy(target, source, sizeof(float) * width * height * maps * samples);
      tensor.Serialize(output_file);
    } */
  }

  std::cout << "\nConfig file output:\n";
  std::cout << ss.str();
  std::cout << "\n";
  
  output_file.close();
  LOGINFO << "Done.";
  LOGEND;
  return 0;
}
