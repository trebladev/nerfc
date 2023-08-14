#include <nerf_loader.h>
#include <filesystem>

void nerf::Loader::load_images(std::string root_dir) {

    // check if dir exists
    std::vector<std::string> sub_dirs = {"test", "train", "val"};
    std::vector<std::string> missing;

    // check if root dir exists
    if (std::filesystem::exists(root_dir)){
        // check if sub dirs exist
        for (const std::string dir : sub_dirs){
            std::filesystem::path path = root_dir + "/" + dir;
            if (!std::filesystem::exists(path)){
                missing.push_back(dir);
            }
        }
        if (!missing.empty()){
            for (const std::string dir : missing){
                std::cout << "Can not find " << dir << " directory!" << std::endl;
            }
        }
        else{
            std::cout << "Load directory: " << root_dir << " successfully"<< std::endl;
        }
    }
    else{
        std::cout << "Can not find root_dir: " << root_dir << "!" << std::endl;
    }

}
