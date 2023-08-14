//
// Created by xuan on 8/14/23.
//

#include <nerf_loader.h>

int main(int argc, char** argv){

    auto* loader = new nerf::Loader();
    loader->load_images("./data/nerf/lego");
    std::cout<< "hello world" << std::endl;

    delete loader;
    return 0;

}