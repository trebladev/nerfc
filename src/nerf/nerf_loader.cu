#include <cuda.h>
#include <cuda_runtime.h>
#include <nerf_loader.h>

#include <filesystem>
#include <random>

void nerf::Loader::load_images() {
  // load images
  count_images();
  // check if dir exists
  std::vector<std::string> sub_dirs = {"test", "train", "val"};
  std::vector<std::string> missing;

  // check if root dir exists
  if (std::filesystem::exists(root_dir)) {
    // check if sub dirs exist
    for (const std::string dir : sub_dirs) {
      std::filesystem::path path = root_dir + "/" + dir;
      if (!std::filesystem::exists(path)) {
        missing.push_back(dir);
      }
    }
    if (!missing.empty()) {
      for (const std::string dir : missing) {
        std::cout << "Can not find " << dir << " directory!" << std::endl;
      }
    } else {
      std::cout << "Load directory: " << root_dir << " successfully"
                << std::endl;
    }
  } else {
    std::cout << "Can not find root_dir: " << root_dir << "!" << std::endl;
  }

  // Read image with sorts
  std::vector<std::string> paths;
  for (const auto& entry :
       std::filesystem::directory_iterator(root_dir + "/" + mode)) {
    paths.push_back(entry.path());
  }

  // sort in alphabetical order
  std::sort(paths.begin(), paths.end());

  tqdm bar(num_image);
  for (const auto& path : paths) {
    cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
    images.push_back(image);
    // std::cout << path << std::endl;
    bar.update();
  }

  H = images[0].rows;
  W = images[0].cols;
  C = images[0].channels();
}

void nerf::Loader::count_images() {
  const std::filesystem::path image_dir = root_dir + "/" + mode;
  num_image = 0;

  for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
    num_image++;
  }

  printf("Found %d images in %s\n", num_image, image_dir.c_str());
}

void nerf::Loader::load_configs(std::string config_path) {
  std::filesystem::path path = config_path;
  if (!std::filesystem::exists(path)) {
    std::cout << "Can not find config file: " << config_path << "!"
              << std::endl;
  } else {
  }

  // Read json file
  Json::Reader reader;
  Json::Value value;
  std::ifstream config(config_path);
  if (!reader.parse(config, value, false)) {
    std::cerr << "parse failed \n";
    return;
  }
  float camera_angle_x = value["camera_angle_x"].asFloat();
  printf("\ncamera_angle_x: %f\n", camera_angle_x);

  Intrinsics.c_x = W / 2.0;
  Intrinsics.c_y = H / 2.0;
  Intrinsics.f_x = 2 * tan(camera_angle_x / 2.0);
  Intrinsics.f_y = 2 * tan(camera_angle_x / 2.0);

  float aspect = W / H;
  float y = H / (2.0 * Intrinsics.f_y);
  projection << 1 / y * aspect, 0, 0, 0, 0, -1 / y, 0, 0, 0, 0,
      -(far + near) / (far - near), -2 * far * near / (far - near), 0, 0, -1, 0;

  for (int i = 0; i < num_image; ++i) {
    Json::Value transform_matrix_json = value["frames"][i]["transform_matrix"];
    Eigen::Matrix4f transform_matrix;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        transform_matrix(i, j) = transform_matrix_json[i][j].asFloat();

    poses.push_back(transform_matrix);
  }

  printf("\nIntrinsics: \n");
  printf("c_x: %f\n", Intrinsics.c_x);
  printf("c_y: %f\n", Intrinsics.c_y);
  printf("f_x: %f\n", Intrinsics.f_x);
  printf("f_y: %f\n", Intrinsics.f_y);
}

void nerf::Loader::generate_rays(int image_idx, int batch_size,
                                 std::vector<Eigen::Vector3f> rays_d,
                                 std::vector<Eigen::Vector3f> rays_o,
                                 std::vector<Eigen::Vector4i> colors) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, H * W - 1);

  for (int i = 0; i < batch_size; ++i) {
    int idx = dis(gen);
    int row = idx / W;
    int col = idx % W;
    Eigen::Vector3f ray_direction =
        (poses[image_idx].block(0, 0, 3, 3) * projection.block(0, 0, 3, 3) *
         Eigen::Vector3f(col, row, 1))
            .normalized();
    Eigen::Vector3f ray_origin = poses[image_idx].block(0, 3, 3, 1);

    rays_d.push_back(ray_direction);
    rays_o.push_back(ray_origin);
    cv::Vec4b rgb = images[image_idx].at<cv::Vec4b>(row, col);
    Eigen::Vector4i eigen_color;
    for (int j = 0; j < 4; ++j) {
      eigen_color(j) = static_cast<int>(rgb[j]);
    }
    colors.push_back(eigen_color);  // 0 - 255

    // printf("The pixel row is %d, col is %d\n", row, col);
    // std::cout << "The pixel rgb is " << rgb << std::endl;
  }
  //   cv::imwrite("r_0.png", images[image_idx]);
}

__global__ void print_thread() {
  printf("Hello world from thread %d\n", threadIdx.x);
}

void nerf::Loader::print() {
  print_thread<<<1, 10>>>();
  cudaDeviceReset();
}
