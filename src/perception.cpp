#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>

// For downsampling
#include <pcl/filters/voxel_grid.h>

// For elapsed time
#include <chrono>

// For ground removal
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>

// For Euclidean Clustering
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

// For image extraction
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Classification
#include <tiny_dnn/tiny_dnn.h>
#include "cone.h"

#include <fstream>

using namespace cv;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

const int nGridX = 8; // Number of grids along x (left - right)
const int nGridZ = 8; // Number of grids along z (front - back)
const float fThresh = 0.05f; // in meter
const int nImageSw =  32;
const int nImageSh = 32;

// For data collection
int record_index = -1;

// Create a network 
network<sequential> net;

inline void print_point(pcl::PointXYZRGB p){
  std::cout << " X: " << p.x
            << " Y: " << p.y 
            << " Z:" << p.z << std::endl;
}

void view_point_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
  pcl::visualization::CloudViewer viewer ("Cloud Viewer");
  viewer.showCloud(cloud);
  while(!viewer.wasStopped()){};
}

void downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample){
  
    // Perform downsampling with voxel
  float fleaf = 0.03f; // Determine the filter size
  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
  vg.setInputCloud(pcloud);
  vg.setLeafSize(fleaf, fleaf, fleaf);
  vg.filter(*sample);
  
  std::cerr << "PointCloud after filtering: " << sample->points.size()
       << " data points \n";
}

void ground_removal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pOut){
  // Gather 3D bounds
  pcl::PointXYZRGB minPoint, maxPoint;
  pcl::getMinMax3D(*pcloud, minPoint, maxPoint);

  /**
  std::cout << "min point: ";
  print_point(minPoint);
  std::cout << "max point: ";
  print_point(maxPoint); **/

  // Calc the size of a grid
  float fGridX = (maxPoint.x - minPoint.x) / nGridX;
  float fGridZ = (maxPoint.z - minPoint.z) / nGridZ;

  // Get absolute values
  fGridX = (fGridX > 0)? fGridX : (-1 * fGridX);
  fGridZ = (fGridZ > 0)? fGridZ : (-1 * fGridZ);

  /**
  std::cout << "GridWidth: " << fGridX 
            << " GridLength: " << fGridZ << std::endl; **/

  pcl::PointXYZ minBox, maxBox; // the 3D point of a box
  // Assign the values for the first grid
  minBox.x = minPoint.x; 
  minBox.y = minPoint.y; 
  minBox.z = minPoint.z;
  maxBox.y = maxPoint.y; 
  maxBox.x = minPoint.x + fGridX;
  maxBox.z = minPoint.z + fGridZ;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr gCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB localMin, localMax;

  // create filter object
  pcl::CropBox<pcl::PointXYZRGB> grid;
  grid.setInputCloud(pcloud);

  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud(gCloud);
  pass.setFilterFieldName("y");

  // Start from min point construct each grid
  for (int xc = 0; xc < nGridX; ++xc){
    for (int zc = 0; zc < nGridZ; ++zc){  
      // For each sector, filter the lower layer
      
      grid.setMin(minBox.getVector4fMap());
      grid.setMax(maxBox.getVector4fMap());
      grid.filter(*gCloud);

      // move the grid point along z
      minBox.z += fGridZ; 
      maxBox.z += fGridZ;

      // Skip if no points
      if (gCloud->points.size() == 0){ continue; }

      // Find minimum point
      pcl::getMinMax3D(*gCloud, localMin, localMax);
      // SKip if this box is very flat
      if (localMax.y - localMin.y < fThresh * 3.6) { continue;}
      // Keep everything between lower thresh hold and 5m from zero plane
      pass.setFilterLimits(localMin.y + fThresh, 1); 
      pass.filter(*gCloud);

      //view_point_cloud(gCloud);
      *pOut += *gCloud;
    }
    // move the grid point along x
    minBox.x += fGridX;
    maxBox.x += fGridX;

    // rest z coord
    minBox.z = minPoint.z;
    maxBox.z = minPoint.z + fGridZ;
  }
}

void euclidean_cluster(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr original,
                      std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& collection){
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);

  // All the things GOD only knows
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(0.1); // with in 10 cm?
  ec.setMinClusterSize(15);
  ec.setMaxClusterSize(1000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  // create filter object
  pcl::CropBox<pcl::PointXYZRGB> grid;
  grid.setInputCloud(original);
  pcl::PointXYZRGB minBox, maxBox;

  // Make point cloud for each cluster
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
      it != cluster_indices.end(); ++it){
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    for (std::vector<int>::const_iterator pit = it->indices.begin(); 
        pit != it->indices.end(); ++pit){
      cloud_cluster->points.push_back (cloud->points[*pit]);
    }
    cloud_cluster->width = cloud_cluster->points.size();
    cloud_cluster->height= 1;
    cloud_cluster->is_dense = true;

    // Cone retrival from original
    // With the bounding box on the downsampled point cloud
    pcl::getMinMax3D(*cloud_cluster, minBox, maxBox);
    grid.setMin(minBox.getVector4fMap());
    grid.setMax(maxBox.getVector4fMap());
    grid.filter(*cloud_cluster);

    collection.push_back(cloud_cluster);
  }
}

float vectorAngle(float x, float y) {
  // add minimal noise to eliminate special cases
  // Would be fine, given that there is a minimum range on LiDAR
  if (x == 0) x = 0.001;
  if (y == 0) y = 0.001;

  float ret = atanf(fabs((float)y/x));
  if ( x > 0 && y < 0) return (M_PI - ret);
  if ( x < 0 && y > 0) return -ret;
  if ( x < 0 && y < 0) return (ret - M_PI);

  return atanf((float)y/x);
}

// This function will record the data
void record_image(cv::Mat& image){
  cv::namedWindow( "Sample", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Sample", image);

  cv::namedWindow( "Gray", cv::WINDOW_AUTOSIZE );
  cv::Mat g;
  cv::cvtColor(image, g, cv::COLOR_RGB2GRAY);
  cv::imshow("Gray", g);

  waitKey(100);

  if (record_index == -1){// ask for starting index to record 
    std::cin.clear();
    std::cout << "Enter the starting index: ";
    std::cin >> record_index;
  }

  char ctype = 0;
  std::cin.clear();
  std::cin.ignore();
  std::cout << "Class of this cone: ";
  std::cin >> ctype;

  if (ctype != '9'){
    // save the image and append the label;
    // Using default locations. (Ain't no time to make it fancy)
    imwrite("res/images/img" + std::to_string(record_index) + ".png", image);
    std::ofstream outfile;
    outfile.open("res/labels.txt", std::ios_base::app);
    outfile << ctype << "\n";
    outfile.close();

    record_index++;
    std::cout << "Saved! " << record_index << std::endl;
  }
}

void project_to_image(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, cv::Mat& image){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr projection (new pcl::PointCloud<pcl::PointXYZRGB>);

  // Calculate the plane perpendicualr to the centre of bounding box
  pcl::PointXYZRGB minBox, maxBox, centrePoint;
  pcl::getMinMax3D(*cloud, minBox, maxBox);

  // Only need x and z
  centrePoint.x = ( minBox.x + maxBox.x ) / 2;
  centrePoint.z = ( minBox.z + maxBox.z ) / 2;
  // Calc rad
  float direction = vectorAngle(centrePoint.x, centrePoint.z);

  // Create a set of planar coeff 
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = cos(direction); // x
  coefficients->values[1] = 0.0;            // y
  coefficients->values[2] = sin(direction); // z
  coefficients->values[3] = 0;

  // Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZRGB> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);
  proj.setInputCloud(cloud);
  proj.setModelCoefficients(coefficients);
  proj.filter(*projection);
  
  // Rotate the plane such that its parallel to the XY plane
  float rotation = direction + M_PI/2; // Some how it just works with this value.

  Eigen::Affine3f transform_plane = Eigen::Affine3f::Identity();
  transform_plane.rotate(Eigen::AngleAxisf (rotation, Eigen::Vector3f::UnitY()));
  // Execute rotation
  pcl::transformPointCloud(*projection, *projection, transform_plane);

  // Calculate the image bound on this flat cloud
  pcl::getMinMax3D(*projection, minBox, maxBox);

  std::cout << "Projection size: " << projection->size() << std::endl;

  // Create black image of size nImageS * nImageS
  image = cv::Mat(nImageSh, nImageSw, CV_8UC3, cv::Scalar(12,12,12)); 

  float ratio_x = nImageSw / (maxBox.x - minBox.x);
  float ratio_y = nImageSh / (maxBox.y - minBox.y); 

  for (std::size_t i = 0; i < projection->points.size (); ++i){
    cv::circle(image, Point(fabs(projection->points[i].x - minBox.x) * ratio_x,
                                 fabs(projection->points[i].y - maxBox.y) * ratio_y), 2, 
                              Scalar(projection->points[i].b, projection->points[i].g, projection->points[i].r), 2);
  }

  //record_image(image);
  // /view_point_cloud(projection);
}

void get_cone_list(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& cluster_collection,
                   std::vector<cone>& cone_collection){
  cv::Mat image; // Cast each cluster to a plane and store as image
  cv::Mat_<uint8_t> gray_scale;
  pcl::PointXYZRGB minp, maxp; // for the cluster's centroid
  cone thisCone;
  for (size_t i = 0; i < cluster_collection.size(); ++i){
    project_to_image(cluster_collection[i], image); 
    pcl::getMinMax3D(*cluster_collection[i], minp, maxp);
    // Convert the colour image to gray scale
    cv::cvtColor(image, gray_scale, cv::COLOR_RGB2GRAY);
    vec_t  vimage;
    std::transform(gray_scale.begin(), gray_scale.end(), std::back_inserter(vimage),
                  [=](uint8_t c) {return c; });

    thisCone.cone_class = net.predict_label(vimage);
    thisCone.cone_accuy = (int)(net.predict(vimage)[thisCone.cone_class]);

    // if the accuracy of the prediction is less than 70 then declare as unknown.
    //if (thisCone.cone_accuy < 70) thisCone.cone_class = -1 ;

    thisCone.x = (maxp.x + minp.x) / 2; // Horizontal 
    thisCone.y = (maxp.y + minp.y) / 2; // Vertical
    thisCone.z = (maxp.z + maxp.z) / 2; // Sideways 
    cone_collection.push_back(thisCone);
  }

}

void initialise(const std::string& weight){
  // Load the network
  // Define the layers
  net << conv(32, 32, 5, 1, 6, padding::same) << activation::tanh()   //in 32x32x1, 5x5conv 6fmap
      << max_pool(32, 32, 6, 2) << activation::tanh()                 //in 32x32x6, 2x2pooling
      << conv(16, 16, 5, 6, 16, padding::same) << activation::tanh()  //in 16x16x6, 5x5conv 16fmaps
      << max_pool(16, 16, 16, 2) << activation::tanh()                //in 16x16x16, 2x2pooling
      << fc(8*8*16, 128) << activation::tanh()
      << fc(128, 64) << activation::tanh()
      << fc(64, 16) << activation::tanh()
      << fc(16, 4) << softmax();                                    //in 16 out 4

  net.load(weight);
}

void detection(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, 
               std::vector<cone>& cone_collection){
  // Start time
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();

  // Downsampled point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sample (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Ground removed & clustering
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr kcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  // Cluster List
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cluster_collection;

  // Downsampling
  downsampling(pcloud, sample);

  // Ground removal
  ground_removal(sample,  kcloud);

  // Euclidean Clustering
  // Clusters will be append into cluster_colletion
  euclidean_cluster(kcloud, pcloud, cluster_collection);

  // Classify each cluster
  get_cone_list(cluster_collection, cone_collection);
  
  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();

  std::cout << "DETECTION TIME = " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count()
            << "[ms]" << std::endl;

  //std::cout << "Cone Collecction Size: "<< cone_collection.size() << std::endl;
  std::cout << "Number of clusters:: " << cluster_collection.size() << std::endl;

  // View result
  //view_point_cloud(pcloud);
  //view_point_cloud(kcloud);
}

int main(int argc, char** argv){

  // The original input/file
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  
  // Read the input pcd file
  std::chrono::steady_clock::time_point begin =\
  std::chrono::steady_clock::now();  
  if (argc > 1){
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *pcloud) == -1){ 
      // load the file
      PCL_ERROR( "Couldn't read file. \n");
    }
  }else{
    std::cout << "No input .pcd file!" << std::endl;
    return (0);
  }  
  
  std::chrono::steady_clock::time_point ending =\
  std::chrono::steady_clock::now();
  
  std::cout << "PCD LOADING TIME = " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(ending - begin).count()
            << "[ms]" << std::endl;

  std::cout << "Loaded "
            << pcloud->width * pcloud ->height
            << " data points from the test.pcd with the following fields: "
            << std::endl;

  // initialise the network 
  initialise("res/AMZ_VODKA");
  
  // List of result
  std::vector<cone> cone_collection;
  // Run detector
  detection(pcloud, cone_collection);
  // print list of detection
  for (size_t i = 0; i < cone_collection.size(); i++){
    std::cout << "Class: " << cone_collection[i].cone_class
              << " X: " << cone_collection[i].x
              << " Y: " << cone_collection[i].y
              << " Z: " << cone_collection[i].z
              << std::endl;
  }

  return (0);  
}