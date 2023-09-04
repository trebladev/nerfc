#include <nerf_loader.h>
#include <filesystem>
#include <tqdm/tqdm.h>

void nerf::Loader::load_images() {

    // load images
    count_images();
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

    tqdm bar(num_image);
    for (const auto & entry : std::filesystem::directory_iterator(root_dir+"/"+mode)){
        std::string path = entry.path();
        cv::Mat image = cv::imread(path);
        images.push_back(image);
//        std::cout << path << std::endl;
        bar.update();
    }

    H = images[0].rows;
    W = images[0].cols;
    C = images[0].channels();

}

void nerf::Loader::count_images() {
    const std::filesystem::path image_dir = root_dir + "/" + mode;
    num_image = 0;

    for (const auto & entry : std::filesystem::directory_iterator(image_dir)){
        num_image++;
    }


    printf("Found %d images in %s\n", num_image, image_dir.c_str());
}

void nerf::Loader::load_configs(std::string config_path)
{
    std::filesystem::path path = config_path;
    if (!std::filesystem::exists(path)){
        std::cout << "Can not find config file: " << config_path << "!" << std::endl;
    }
    else {

    }

    Json::Reader reader;
    Json::Value value;
    std::ifstream config(config_path);
    if (!reader.parse(config, value, false)){
        std::cerr << "parse failed \n";
		return;
    }
    std::string project_name = value["name"].asString();
    printf("\nProject name: %s\n", project_name.c_str());

}
