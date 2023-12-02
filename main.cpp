//
// Created by xuan on 8/14/23.
//

#include <nerf_loader.h>
#include <utils.h>

#include <argparse.hpp>

int main(int argc, char** argv) {
  argparse::ArgumentParser arg("Config of nerf.");
  arg.add_argument("-w", "--work_space")
      .required()
      .help("The directory of nerf results");
  arg.add_argument("-m", "--mode").required();
  arg.add_argument("-c", "--config").required();
  arg.add_argument("-data").required();

  auto* loader = new nerf::Loader("./data/nerf/chair", "train");
  loader->load_images();
  loader->load_configs("./data/nerf/chair/transforms_train.json");
  loader->generate_rays(0, 512, std::vector<Eigen::Vector3f>(),
                        std::vector<Eigen::Vector3f>(),
                        std::vector<Eigen::Vector4i>());
  //    std::cout<< "hello world" << std::endl;
  //   loader->print();

  delete loader;

  // // Free the gpu memory
  // for (const VariableInfo& variable : uploadedVariables){
  //     CHECK(cudaFree(variable.data));
  // }
  return 0;
}