//
// Created by xuan on 8/14/23.
//

#include <nerf_loader.h>

int main(int argc, char** argv){

    auto* loader = new nerf::Loader("./data/nerf/chair", "train");
    loader->load_images();
    loader->load_configs("./conf/nerf_synthetic.json");
//    std::cout<< "hello world" << std::endl;

    delete loader;
    return 0;

}