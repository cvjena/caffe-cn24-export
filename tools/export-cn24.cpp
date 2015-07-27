#include <cstring>
#include <cstdlib>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
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

  NetParameter net_param;
  string input_filename(argv[1]);
  string output_filename(argv[2]);
  std::ofstream output_file(output_filename, std::ios::out | std::ios::binary);
  
  if (!ReadProtoFromBinaryFile(input_filename, &net_param)) {
    LOG(ERROR) << "Failed to parse input binary file as NetParameter: "
               << input_filename;
    return 2;
  }

  LOG(ERROR) << "Processing model: " << net_param.name();
  LOG(ERROR) << "Layers in this model: " << net_param.layer_size();

  for(int layer_id = 0; layer_id < net_param.layer_size(); layer_id++) {
    const LayerParameter& layer_param = net_param.layer(layer_id);
    LOG(ERROR) << "Processing layer " << layer_param.name();
    for (int i = 0; i < layer_param.blobs_size(); ++i) {
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
    }
  }
  
  output_file.close();
  LOG(ERROR) << "Done.";
  LOGEND;
  return 0;
}
